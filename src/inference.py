"""
src/inference.py
Module suy luận thống nhất cho cả 3 mô hình:
- Fine-tuned PhoBERT (`models/cmc_phobert_sentiment`)
- TF-IDF + Logistic Regression (`models/model_tfidf_lr.joblib`)
- TF-IDF + LinearSVC calibrated (`models/model_tfidf_svm.joblib`)
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import joblib
import numpy as np

from src.anonymize import anonymize_text
from src.preprocess import clean_text, segment_for_phobert
from src.topics import classify_topic

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HF_CACHE_DIR = ROOT / ".hf_cache"

LABELS: List[str] = ["Negative", "Neutral", "Positive"]
ID2LABEL: Dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2VI: Dict[str, str] = {
    "Negative": "Tiêu cực",
    "Neutral": "Bình thường",
    "Positive": "Tích cực",
}


@dataclass
class PredictionResult:
    raw_text: str
    anonymized_text: str
    cleaned_text: str
    label: str
    label_vi: str
    confidence: float
    probabilities: Dict[str, float]
    topic: str
    backend: str
    latency_ms: float
    top_features: List[Tuple[str, float]] = field(default_factory=list)


def _phobert_weights_ready(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    has_config = (model_dir / "config.json").exists()
    has_weights = any(
        (model_dir / fname).exists()
        for fname in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


class SentimentEngine:
    """Bộ suy luận cảm xúc thống nhất, tuyệt đối không dùng fallback từ khóa cứng."""

    def __init__(
        self,
        backend: str = "phobert",
        models_dir: Optional[Path] = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.requested_backend = backend
        self.backend = backend
        self.backend_status = ""
        self._pipeline = None
        self._tokenizer = None
        self._phobert_model = None
        self._device = "cpu"
        self._phobert_max_len = 128
        self._phobert_use_seg = False
        self._id2label = dict(ID2LABEL)

        self._load_backend(backend)

    def _load_backend(self, backend: str) -> None:
        if backend == "phobert":
            phobert_dir = self.models_dir / "cmc_phobert_sentiment"
            if _phobert_weights_ready(phobert_dir):
                try:
                    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
                    import torch
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer

                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(phobert_dir), local_files_only=True
                    )
                    self._phobert_model = AutoModelForSequenceClassification.from_pretrained(
                        str(phobert_dir), local_files_only=True
                    )
                    self._phobert_model.to(self._device)
                    self._phobert_model.eval()

                    summary_path = phobert_dir / "training_summary.json"
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        self._phobert_max_len = int(summary.get("max_length", 128))
                        self._phobert_use_seg = bool(summary.get("word_segmentation", False))

                    cfg_id2label = getattr(self._phobert_model.config, "id2label", None)
                    if isinstance(cfg_id2label, dict) and cfg_id2label:
                        parsed = {int(k): str(v) for k, v in cfg_id2label.items()}
                        if set(parsed.values()) == set(LABELS):
                            self._id2label = parsed

                    self.backend = "phobert"
                    self.backend_status = (
                        f"Fine-tuned PhoBERT (local) [{self._device.upper()}]"
                    )
                    return
                except Exception as exc:
                    warnings.warn(
                        f"Không thể nạp PhoBERT từ {phobert_dir}: {exc}. "
                        "Chuyển sang mô hình ML TF-IDF + Logistic Regression.",
                        RuntimeWarning,
                    )
            else:
                warnings.warn(
                    f"Chưa có trọng số PhoBERT tại {phobert_dir}. "
                    "Chuyển sang mô hình ML TF-IDF + Logistic Regression.",
                    RuntimeWarning,
                )
            backend = "tfidf_lr"

        if backend == "tfidf_lr":
            lr_path = self.models_dir / "model_tfidf_lr.joblib"
            if not lr_path.exists():
                raise FileNotFoundError(f"Không tìm thấy mô hình: {lr_path}")
            self._pipeline = joblib.load(lr_path)
            self.backend = "tfidf_lr"
            self.backend_status = "TF-IDF + Logistic Regression (model_tfidf_lr.joblib)"
            return

        if backend == "tfidf_svm":
            svm_path = self.models_dir / "model_tfidf_svm.joblib"
            if not svm_path.exists():
                raise FileNotFoundError(f"Không tìm thấy mô hình: {svm_path}")
            self._pipeline = joblib.load(svm_path)
            self.backend = "tfidf_svm"
            self.backend_status = "TF-IDF (Word+Char) + LinearSVC (model_tfidf_svm.joblib)"
            return

        raise ValueError(f"Backend không hợp lệ: {backend}")

    def _explain_sklearn(self, cleaned: str, pred_class_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._pipeline is None or not cleaned:
            return []
        try:
            named_steps = getattr(self._pipeline, "named_steps", {})
            vectorizer = named_steps.get("tfidf") or named_steps.get("features")
            clf = named_steps.get("clf")
            if vectorizer is None or clf is None:
                return []

            X = vectorizer.transform([cleaned])
            indices = X.nonzero()[1]
            if len(indices) == 0:
                return []

            feature_names = vectorizer.get_feature_names_out()
            data_vals = X.toarray()[0]

            if hasattr(clf, "coef_"):
                weights = clf.coef_[pred_class_idx]
                contribs = [(str(feature_names[i]), round(float(data_vals[i] * weights[i]), 4)) for i in indices]
                contribs.sort(key=lambda x: abs(x[1]), reverse=True)
                return contribs[:top_k]

            # CalibratedClassifierCV (e.g. LinearSVC)
            if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
                coefs = [
                    cc.estimator.coef_[pred_class_idx]
                    for cc in clf.calibrated_classifiers_
                    if hasattr(cc, "estimator") and hasattr(cc.estimator, "coef_")
                ]
                if coefs:
                    avg_w = np.mean(coefs, axis=0)
                    contribs = [(str(feature_names[i]), round(float(data_vals[i] * avg_w[i]), 4)) for i in indices]
                    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
                    return contribs[:top_k]

            return [(str(feature_names[i]), round(float(data_vals[i]), 4)) for i in indices[:top_k]]
        except Exception:
            return []

    def _predict_probs_batch(self, cleaned_texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        if not cleaned_texts:
            return []

        if self.backend == "phobert":
            import torch

            model_inputs = [
                segment_for_phobert(t, use_segmentation=self._phobert_use_seg)
                for t in cleaned_texts
            ]
            # Sắp xếp theo độ dài để giảm padding thừa khi suy luận batch trên CPU, rồi khôi phục thứ tự
            order = sorted(range(len(model_inputs)), key=lambda i: len(model_inputs[i]))
            sorted_inputs = [model_inputs[i] for i in order]
            sorted_probs: List[Dict[str, float]] = []

            for start in range(0, len(sorted_inputs), batch_size):
                batch = sorted_inputs[start : start + batch_size]
                enc = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._phobert_max_len,
                    padding=True,
                ).to(self._device)
                with torch.no_grad():
                    logits = self._phobert_model(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    prob_dict = {
                        self._id2label[idx]: float(row[idx])
                        for idx in range(len(row))
                    }
                    sorted_probs.append({lbl: prob_dict.get(lbl, 0.0) for lbl in LABELS})

            restored: List[Optional[Dict[str, float]]] = [None] * len(model_inputs)
            for orig_idx, p in zip(order, sorted_probs):
                restored[orig_idx] = p
            return [p for p in restored if p is not None]

        # sklearn pipelines
        raw_probs = self._pipeline.predict_proba(cleaned_texts)
        classes = [int(c) for c in self._pipeline.classes_]
        out: List[Dict[str, float]] = []
        for row in raw_probs:
            prob_dict = {ID2LABEL[classes[i]]: float(row[i]) for i in range(len(classes))}
            out.append({lbl: prob_dict.get(lbl, 0.0) for lbl in LABELS})
        return out

    def predict(self, text: str) -> PredictionResult:
        t0 = time.perf_counter()
        raw = str(text or "")
        anon = anonymize_text(raw)
        cleaned = clean_text(raw)
        topic = classify_topic(anon or raw)

        if len(cleaned) < 2:
            probs = {"Negative": 0.05, "Neutral": 0.90, "Positive": 0.05}
            elapsed = (time.perf_counter() - t0) * 1000.0
            return PredictionResult(
                raw_text=raw,
                anonymized_text=anon,
                cleaned_text=cleaned,
                label="Neutral",
                label_vi=LABEL2VI["Neutral"],
                confidence=0.90,
                probabilities=probs,
                topic=topic,
                backend=self.backend,
                latency_ms=round(elapsed, 2),
                top_features=[],
            )

        probs = self._predict_probs_batch([cleaned], batch_size=1)[0]
        best_label = max(LABELS, key=lambda lbl: probs[lbl])
        conf = round(float(probs[best_label]), 4)
        rounded_probs = {lbl: round(float(probs[lbl]), 4) for lbl in LABELS}

        top_features: List[Tuple[str, float]] = []
        if self.backend in ("tfidf_lr", "tfidf_svm") and self._pipeline is not None:
            classes = [int(c) for c in self._pipeline.classes_]
            label_idx = LABELS.index(best_label)
            if label_idx in classes:
                pred_col = classes.index(label_idx)
                top_features = self._explain_sklearn(cleaned, pred_col, top_k=5)

        elapsed = (time.perf_counter() - t0) * 1000.0
        return PredictionResult(
            raw_text=raw,
            anonymized_text=anon,
            cleaned_text=cleaned,
            label=best_label,
            label_vi=LABEL2VI[best_label],
            confidence=conf,
            probabilities=rounded_probs,
            topic=topic,
            backend=self.backend,
            latency_ms=round(elapsed, 2),
            top_features=top_features,
        )

    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[PredictionResult]:
        t0 = time.perf_counter()
        raws = [str(t or "") for t in texts]
        anons = [anonymize_text(r) for r in raws]
        cleaneds = [clean_text(r) for r in raws]
        topics = [classify_topic(a or r) for a, r in zip(anons, raws)]

        valid_indices = [i for i, c in enumerate(cleaneds) if len(c) >= 2]
        valid_texts = [cleaneds[i] for i in valid_indices]
        valid_probs = self._predict_probs_batch(valid_texts, batch_size=batch_size)

        probs_map: Dict[int, Dict[str, float]] = {
            idx: p for idx, p in zip(valid_indices, valid_probs)
        }
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = round(total_ms / max(len(raws), 1), 2)

        results: List[PredictionResult] = []
        for i, raw in enumerate(raws):
            if i not in probs_map:
                probs = {"Negative": 0.05, "Neutral": 0.90, "Positive": 0.05}
                best = "Neutral"
                conf = 0.90
            else:
                probs = {lbl: round(float(probs_map[i][lbl]), 4) for lbl in LABELS}
                best = max(LABELS, key=lambda lbl: probs[lbl])
                conf = round(float(probs[best]), 4)

            results.append(
                PredictionResult(
                    raw_text=raw,
                    anonymized_text=anons[i],
                    cleaned_text=cleaneds[i],
                    label=best,
                    label_vi=LABEL2VI[best],
                    confidence=conf,
                    probabilities=probs,
                    topic=topics[i],
                    backend=self.backend,
                    latency_ms=per_item_ms,
                    top_features=[],
                )
            )
        return results
