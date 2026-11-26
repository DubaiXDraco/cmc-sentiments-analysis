#!/usr/bin/env python3
"""
scripts/07_package_submission.py
Đóng gói dự án hoàn chỉnh thành 2 file `.zip`:
1. `cmc-student-sentiment-final-lite.zip`: gọn (< 15 MB) dùng để nộp LMS / GitHub (bỏ .venv, .hf_cache, weight PhoBERT lớn)
2. `cmc-student-sentiment-final-full.zip`: đầy đủ chạy offline ngay cả với PhoBERT (`model.safetensors` lưu ở chế độ ZIP_STORED)
Tuyệt đối loại bỏ `.venv`, `.hf_cache`, `__pycache__`, `.pytest_cache`, `.git`, và mọi file `dataset_*.json`.
"""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = ROOT.parent

LITE_ZIP = SCRATCH_DIR / "cmc-student-sentiment-final-lite.zip"
FULL_ZIP = SCRATCH_DIR / "cmc-student-sentiment-final-full.zip"

EXCLUDE_DIRS = {".venv", ".hf_cache", "__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}
LARGE_WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth"}


def should_skip(rel_parts: tuple, filename: str, include_large_weights: bool) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel_parts):
        return True
    if filename.startswith("dataset_") and filename.endswith(".json"):
        return True
    if filename.endswith((".pyc", ".pyo")):
        return True
    if not include_large_weights and Path(filename).suffix.lower() in LARGE_WEIGHT_EXTS:
        return True
    return False


def build_zip(dest_zip: Path, include_large_weights: bool) -> tuple:
    if dest_zip.exists():
        dest_zip.unlink()

    file_count = 0
    with zipfile.ZipFile(dest_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if should_skip(rel.parts[:-1], path.name, include_large_weights=include_large_weights):
                continue
            arcname = str(Path("cmc-student-sentiment-final") / rel)
            is_large_bin = path.suffix.lower() in LARGE_WEIGHT_EXTS
            compress = zipfile.ZIP_STORED if is_large_bin else zipfile.ZIP_DEFLATED
            zf.write(path, arcname=arcname, compress_type=compress)
            file_count += 1

    # Kiểm tra toàn vẹn zip
    with zipfile.ZipFile(dest_zip, mode="r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Zip bị lỗi tại file: {bad}")
        names = zf.namelist()
        assert not any(".venv/" in n or ".hf_cache/" in n for n in names)
        assert not any("dataset_" in n and n.endswith(".json") for n in names)

    size_mb = round(dest_zip.stat().st_size / (1024 * 1024), 2)
    return file_count, size_mb


def main() -> None:
    lite_cnt, lite_mb = build_zip(LITE_ZIP, include_large_weights=False)
    print(f"✓ Lite Zip: {LITE_ZIP} ({lite_cnt} files, {lite_mb} MB)")
    full_cnt, full_mb = build_zip(FULL_ZIP, include_large_weights=True)
    print(f"✓ Full Zip: {FULL_ZIP} ({full_cnt} files, {full_mb} MB)")


if __name__ == "__main__":
    main()
