"""
src/anonymize.py
Ẩn danh hóa định danh và thông tin cá nhân (PII) trước khi lưu trữ hoặc huấn luyện.
"""

import hashlib
import re
from typing import Optional

_SALT = "cmc_student_sentiment_2026_privacy_salt"

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w\-]+\.[\w.\-]+\b", flags=re.IGNORECASE)
IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?::\d{1,5})?\b"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?84[\s.\-]?|0)[235789](?:\d{7,9}|(?:\d{1,2}[\s.\-]\d{3,4}[\s.\-]\d{3,4}))(?!\d)"
)
STUDENT_ID_RE = re.compile(r"\b[a-z]{2,4}2[0-6]\d{4,6}\b", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.\-]+")
CONTACT_NAME_RE = re.compile(
    r"((?:nhắn tin|ib|inb|liên hệ|vợ tôi|fb|facebook)[:\s]+)"
    r"([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,3})",
    flags=re.UNICODE,
)


def anonymize_id(raw_id: Optional[str]) -> str:
    """Băm SHA-256 có salt thành ID ẩn danh dạng `cmc_<12hex>`."""
    raw = str(raw_id or "").strip()
    digest = hashlib.sha256(f"{_SALT}:{raw}".encode("utf-8")).hexdigest()[:12]
    return f"cmc_{digest}"


def anonymize_text(text: Optional[str]) -> str:
    """Ẩn danh hóa URL, email, IP, số điện thoại, mã sinh viên, @mention và tên sau cụm liên hệ."""
    if not isinstance(text, str) or not text.strip():
        return ""

    out = text
    out = URL_RE.sub("", out)
    out = EMAIL_RE.sub("[EMAIL]", out)
    out = IPV4_RE.sub("[IP]", out)
    out = PHONE_RE.sub("[PHONE]", out)
    out = STUDENT_ID_RE.sub("[STUDENT_ID]", out)
    out = MENTION_RE.sub("[USER]", out)
    out = CONTACT_NAME_RE.sub(r"\1[NAME]", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def contains_pii(text: Optional[str]) -> bool:
    """Kiểm tra xem chuỗi văn bản còn sót PII (URL, email, IP, SĐT, MSSV, @mention) hay không."""
    if not isinstance(text, str) or not text:
        return False
    checks = (URL_RE, EMAIL_RE, IPV4_RE, PHONE_RE, STUDENT_ID_RE, MENTION_RE)
    return any(pattern.search(text) is not None for pattern in checks)
