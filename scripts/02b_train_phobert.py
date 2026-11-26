#!/usr/bin/env python3
"""
scripts/02b_train_phobert.py
Tuần 2: Fine-tune mô hình PhoBERT (`vinai/phobert-base-v2`) trên dữ liệu thật CMC
với seed cố định, class-weighted CrossEntropyLoss xử lý mất cân bằng,
tối ưu tốc độ CPU bằng length-sorted batching + frozen-prefix hidden-state caching
(tương đương toán học với đóng băng tầng đáy, kiểm chứng bằng torch.allclose),
và lưu checkpoint có Val Macro-F1 cao nhất.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import has_word_segmenter, segment_for_phobert

HF_CACHE_DIR = ROOT / ".hf_cache"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models" / "cmc_phobert_sentiment"

LABEL2ID: Dict[str, int] = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL: Dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def build_length_sorted_batches(
    texts: List[str],
    labels: List[int],
    tokenizer,
    max_length: int,
    batch_size: int,
) -> List[Dict[str, torch.Tensor]]:
    """Mã hóa và gom batch đã sắp xếp theo độ dài token để tối thiểu hóa padding thừa."""
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    order = sorted(range(len(texts)), key=lambda i: len(enc["input_ids"][i]))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        max_len = max(len(enc["input_ids"][i]) for i in idxs)
        batch_ids, batch_mask, batch_lbl = [], [], []
        for i in idxs:
            ids = enc["input_ids"][i]
            mask = enc["attention_mask"][i]
            pad_n = max_len - len(ids)
            batch_ids.append(ids + [pad_id] * pad_n)
            batch_mask.append(mask + [0] * pad_n)
            batch_lbl.append(labels[i])
        batches.append(
            {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
                "labels": torch.tensor(batch_lbl, dtype=torch.long),
            }
        )
    return batches


def configure_trainable_layers(model, unfreeze_layers: int) -> int:
    """Đóng băng embeddings & các tầng encoder dưới; chỉ mở top `unfreeze_layers` + head."""
    if unfreeze_layers < 0 or unfreeze_layers >= len(model.roberta.encoder.layer):
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True

        total_layers = len(model.roberta.encoder.layer)
        start_idx = max(0, total_layers - unfreeze_layers)
        for layer_idx in range(start_idx, total_layers):
            for p in model.roberta.encoder.layer[layer_idx].parameters():
                p.requires_grad = True

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_frozen_prefix_cache(
    model,
    raw_batches: List[Dict[str, torch.Tensor]],
    split_idx: int,
    device: str,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    """Chạy 1 lượt qua embeddings + tầng đóng băng [0..split_idx-1] ở chế độ eval()."""
    model.eval()
    cached: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = []
    with torch.no_grad():
        for b in raw_batches:
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            labels = b["labels"].to(device)

            emb = model.roberta.embeddings(input_ids=input_ids)
            attn_mask_4d, _ = model.roberta._create_attention_masks(
                attention_mask=attention_mask,
                encoder_attention_mask=None,
                embedding_output=emb,
                encoder_hidden_states=None,
                cache=None,
            )
            h = emb
            for layer_idx in range(split_idx):
                h = model.roberta.encoder.layer[layer_idx](
                    h,
                    attn_mask_4d,
                    None,
                    encoder_attention_mask=None,
                    cache=None,
                )
            cached.append((h.detach(), attn_mask_4d, labels))
    return cached


def forward_top_layers(model, h: torch.Tensor, attn_mask_4d: Optional[torch.Tensor], split_idx: int) -> torch.Tensor:
    out = h
    for layer_idx in range(split_idx, len(model.roberta.encoder.layer)):
        out = model.roberta.encoder.layer[layer_idx](
            out,
            attn_mask_4d,
            None,
            encoder_attention_mask=None,
            cache=None,
        )
    return model.classifier(out)


def evaluate_cached(model, cached_val, loss_fn, split_idx: int) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for h, attn_mask_4d, labels in cached_val:
            logits = forward_top_layers(model, h, attn_mask_4d, split_idx)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            preds_all.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(len(labels_all), 1)
    return avg_loss, float(accuracy_score(labels_all, preds_all)), float(f1_score(labels_all, preds_all, average="macro"))


def evaluate_full(model, loader, loss_fn, device: str) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += float(loss.item()) * len(labels)
            preds_all.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(len(labels_all), 1)
    return avg_loss, float(accuracy_score(labels_all, preds_all)), float(f1_score(labels_all, preds_all, average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT trên tập Train/Val CMC.")
    parser.add_argument("--base-model", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        cpu_threads = max(2, min(10, (os.cpu_count() or 4)))
        torch.set_num_threads(cpu_threads)

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(DATA_DIR / "val.csv", encoding="utf-8-sig")

    use_seg = has_word_segmenter()
    train_texts = [segment_for_phobert(t, use_segmentation=use_seg) for t in train_df["text_clean"].fillna("")]
    val_texts = [segment_for_phobert(t, use_segmentation=use_seg) for t in val_df["text_clean"].fillna("")]
    train_labels = train_df["label"].map(LABEL2ID).astype(int).tolist()
    val_labels = val_df["label"].map(LABEL2ID).astype(int).tolist()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    total_layers = len(model.roberta.encoder.layer)
    trainable_params = configure_trainable_layers(model, args.unfreeze_layers)
    total_params = int(sum(p.numel() for p in model.parameters()))
    model.to(device)

    # Class-weighted CrossEntropyLoss xử lý mất cân bằng dữ liệu
    counts = np.bincount(train_labels, minlength=3).astype(float)
    weights = len(train_labels) / (3.0 * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    use_prefix_cache = 0 < args.unfreeze_layers < total_layers
    split_idx = total_layers - args.unfreeze_layers if use_prefix_cache else 0

    print("=" * 64, flush=True)
    print(f"FINE-TUNE PHOBERT ({args.base_model}) TRÊN {device.upper()}", flush=True)
    print("=" * 64, flush=True)
    print(f"- Train: {len(train_texts)} | Val: {len(val_texts)} | word_segmentation={use_seg}", flush=True)
    print(f"- Trainable params: {trainable_params:,} / {total_params:,} (unfreeze_layers={args.unfreeze_layers})", flush=True)
    print(f"- Class weights: {[round(float(w), 4) for w in weights]}", flush=True)

    t_start = time.perf_counter()
    if use_prefix_cache:
        raw_train_batches = build_length_sorted_batches(train_texts, train_labels, tokenizer, args.max_length, args.batch_size)
        raw_val_batches = build_length_sorted_batches(val_texts, val_labels, tokenizer, args.max_length, args.batch_size)

        t_cache = time.perf_counter()
        cached_train = compute_frozen_prefix_cache(model, raw_train_batches, split_idx, device)
        cached_val = compute_frozen_prefix_cache(model, raw_val_batches, split_idx, device)
        cache_sec = round(time.perf_counter() - t_cache, 2)
        print(f"- Cached frozen-prefix hidden states (layers 0..{split_idx-1}) in {cache_sec}s", flush=True)

        # Kiểm chứng tương đương toán học giữa forward từ prefix cache và forward đầy đủ
        model.eval()
        with torch.no_grad():
            b0 = raw_val_batches[0]
            full_logits = model(
                input_ids=b0["input_ids"].to(device),
                attention_mask=b0["attention_mask"].to(device),
            ).logits
            cached_logits = forward_top_layers(model, cached_val[0][0], cached_val[0][1], split_idx)
            max_diff = float((full_logits - cached_logits).abs().max().item())
            assert torch.allclose(full_logits, cached_logits, atol=1e-4), (
                f"Prefix cache không khớp full forward (max_diff={max_diff})"
            )
        print(f"- Equivalence check passed: max_diff={max_diff:.2e} <= 1e-4", flush=True)
        n_train_batches = len(cached_train)
    else:
        train_ds = SentimentDataset(train_texts, train_labels, tokenizer, args.max_length)
        val_ds = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        g = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collator)
        n_train_batches = len(train_loader)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = max(1, n_train_batches * args.epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        if use_prefix_cache:
            model.eval()
            for layer_idx in range(split_idx, total_layers):
                model.roberta.encoder.layer[layer_idx].train()
            model.classifier.train()

            epoch_batches = list(cached_train)
            random.Random(args.seed + epoch).shuffle(epoch_batches)
            running_loss = 0.0
            for h, attn_mask_4d, labels in epoch_batches:
                optimizer.zero_grad()
                logits = forward_top_layers(model, h, attn_mask_4d, split_idx)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.item()) * len(labels)

            train_loss = running_loss / max(len(train_texts), 1)
            val_loss, val_acc, val_f1 = evaluate_cached(model, cached_val, loss_fn, split_idx)
        else:
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.item()) * len(labels)

            train_loss = running_loss / max(len(train_texts), 1)
            val_loss, val_acc, val_f1 = evaluate_full(model, val_loader, loss_fn, device)

        ep_sec = round(time.perf_counter() - ep_t0, 2)
        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "elapsed_sec": ep_sec,
        }
        history.append(rec)
        print(
            f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} ({ep_sec}s)",
            flush=True,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    total_sec = round(time.perf_counter() - t_start, 2)
    summary = {
        "base_model": args.base_model,
        "seed": args.seed,
        "device": device,
        "optimization": "length_sorted_frozen_prefix_cache" if use_prefix_cache else "full_forward",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "unfreeze_layers": args.unfreeze_layers,
        "word_segmentation": use_seg,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "class_weights": [round(float(w), 4) for w in weights],
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "total_train_sec": total_sec,
        "history": history,
    }
    (OUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✓ Đã lưu mô hình tốt nhất (Epoch {best_epoch}, Val Macro-F1={best_val_f1:.4f}) tại {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
