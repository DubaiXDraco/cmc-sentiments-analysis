"""
src/preprocess.py
Hàm tiền xử lý thống nhất dùng chung cho thu thập, huấn luyện, pipeline hàng ngày, demo và test.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

from src.anonymize import anonymize_text

try:
    from pyvi import ViTokenizer as _ViTokenizer
except ImportError:
    _ViTokenizer = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEENCODE_PATH = ROOT / "data" / "teencode.json"

_PLACEHOLDER_RE = re.compile(
    r"\[(?:PHONE|EMAIL|STUDENT_ID|IP|USER|NAME)\]",
    flags=re.IGNORECASE,
)
_REPEATED_CHAR_RE = re.compile(
    r"([a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ])\1{2,}"
)
_KEEP_CHARS_RE = re.compile(
    r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]"
)
_SAFE_K_RE = re.compile(r"(?<!\w)k(?!\d|\w)")

INTENSIFIER_PATTERN = r"\b(?:quá|lắm|luôn|nữa|thật|ghê|vãi|kinh khủng|cực kỳ|cực)\b"

PREFIX_PATTERNS = [
    r"^mọi người ơi[,\s]*",
    r"^cho mình hỏi[,\s]*",
    r"^cá nhân mình thấy[,\s]*",
    r"^cá nhân mình[,\s]*",
    r"^thật sự[,\s]*",
    r"^mọi người[,\s]*",
]

SUFFIX_PATTERNS = [
    r"\s+ạ$",
    r"\s+nhé$",
    r"\s+nha$",
    r"\s+mọi người$",
    r"\s+với$",
    r"\s+không$",
    r"\s+thật sự$",
    r"\s*!+$",
    r"\s*\?+$",
]


def load_teencode_dict(path: Optional[Path] = None) -> Dict[str, str]:
    target = path or DEFAULT_TEENCODE_PATH
    if not target.exists():
        return {}
    raw = json.loads(target.read_text(encoding="utf-8"))
    return {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in raw.items()
        if str(k).strip()
    }


_TEENCODE_DICT: Dict[str, str] = load_teencode_dict()
if _TEENCODE_DICT:
    _SORTED_KEYS = sorted(_TEENCODE_DICT.keys(), key=len, reverse=True)
    _TEENCODE_RE = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in _SORTED_KEYS) + r")(?!\w)"
    )
else:
    _TEENCODE_RE = None


def apply_teencode(text: str) -> str:
    """Thay thế teencode theo ranh giới từ và bảo vệ các cụm k3/k4/k5."""
    if not text:
        return ""
    out = _SAFE_K_RE.sub("không", text)
    if _TEENCODE_RE is not None:
        out = _TEENCODE_RE.sub(lambda m: _TEENCODE_DICT.get(m.group(0), m.group(0)), out)
    return re.sub(r"\s+", " ", out).strip()


def clean_text(text: Optional[str]) -> str:
    """Làm sạch, ẩn danh PII và chuẩn hóa teencode tiếng Việt."""
    if not isinstance(text, str) or not text.strip():
        return ""

    t = unicodedata.normalize("NFC", text)
    t = anonymize_text(t)
    t = _PLACEHOLDER_RE.sub(" ", t)
    t = re.sub(r"#\w+", " ", t)
    t = t.lower()
    t = _REPEATED_CHAR_RE.sub(r"\1", t)
    t = _KEEP_CHARS_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = apply_teencode(t)
    return re.sub(r"\s+", " ", t).strip()


def has_word_segmenter() -> bool:
    return _ViTokenizer is not None


def segment_for_phobert(text: str, use_segmentation: Optional[bool] = None) -> str:
    """Tách từ tiếng Việt cho PhoBERT nếu pyvi khả dụng và được bật."""
    if not text:
        return ""
    if use_segmentation is False:
        return text
    if _ViTokenizer is not None:
        return _ViTokenizer.tokenize(text)
    return text


def make_core_text(text: Optional[str]) -> str:
    """
    Tạo khóa câu gốc (`core_text`) dùng để loại trùng và chia GroupShuffleSplit
    nhằm ngăn rò rỉ dữ liệu giữa các tập train / val / test.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = clean_text(text)
    t = re.sub(r"\bnhà trường\b", "trường", t)

    prev = None
    while prev != t:
        prev = t
        for p in PREFIX_PATTERNS:
            t = re.sub(p, "", t, flags=re.IGNORECASE).strip()

    prev = None
    while prev != t:
        prev = t
        for p in SUFFIX_PATTERNS:
            t = re.sub(p, "", t, flags=re.IGNORECASE).strip()

    t = re.sub(INTENSIFIER_PATTERN, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def make_loose_key(text: Optional[str]) -> str:
    """
    Khóa lỏng (bỏ dấu, bỏ từ đệm/phủ định) để kiểm tra bảo thủ trùng lặp
    trên bộ kiểm tra thử thách độc lập; không dùng để chia tập huấn luyện.
    """
    t = make_core_text(text)
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").replace("đ", "d")
    t = re.sub(r"\b(a|nhe|nha|voi|ko|khong)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()
