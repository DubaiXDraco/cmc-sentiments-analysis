"""
src/topics.py
Phân loại chủ đề bài viết / bình luận sinh viên CMC thành 6 nhóm rõ ràng.
"""

import re
from typing import Dict, List

from src.preprocess import clean_text

TOPICS: List[str] = [
    "Hoạt động & CLB",
    "Học vụ & Giảng viên",
    "Cơ sở vật chất & CNTT",
    "Đời sống & Nhà trọ",
    "Tìm đồ & Hỗ trợ",
    "Giao lưu & Khác",
]

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Hoạt động & CLB": [
        "câu lạc bộ", "clb", "casting", "tuyển thành viên", "gen", "fest", "10fest",
        "sự kiện", "ban nhạc", "nhảy", "múa", "khởi nghiệp", "hatch", "gim",
        "the ix", "okc", "rebels", "esl", "cume", "ctxh", "esport", "ctf",
        "giải đấu", "tình nguyện", "club day", "tryout", "bóng rổ", "ambassadors",
    ],
    "Học vụ & Giảng viên": [
        "giảng viên", "thầy", "cô", "môn", "học phần", "tín chỉ", "lịch thi",
        "lịch học", "bài tập", "hạn nộp bài", "điểm", "qua môn", "trượt môn",
        "công tác sinh viên", "học bổng", "thực tập", "tiếng anh", "toán",
        "học phí", "thời khóa biểu", "vstep", "giữa kỳ", "cuối kỳ", "học kỳ",
        "đào tạo", "cấu trúc dữ liệu", "trí tuệ nhân tạo", "seminar", "slide",
    ],
    "Cơ sở vật chất & CNTT": [
        "cơ sở vật chất", "phòng thực hành", "phòng học", "phòng lab", "lab",
        "cơ sở 1", "cơ sở 2", "cơ sở mới", "wifi", "mạng", "điều hòa",
        "máy lạnh", "thư viện", "thang máy", "itest", "đăng ký môn học",
        "cổng thông tin", "server", "hệ thống", "máy tính", "tầng",
    ],
    "Đời sống & Nhà trọ": [
        "trọ", "phòng trọ", "roommate", "roomate", "ở ghép", "phòng ở",
        "tiền phòng", "vệ sinh khép kín", "vạn phúc", "hà đông", "căng tin",
        "đồ ăn", "gửi xe", "xe điện", "ship", "billiards", "ký túc xá",
        "quán cà phê", "áo khoác",
    ],
    "Tìm đồ & Hỗ trợ": [
        "để quên", "nhặt được", "đánh rơi", "tìm lại", "thẻ sinh viên",
        "móc khóa", "móc khoá", "chìa khóa", "chìa khoá", "ipad",
        "sạc dự phòng", "điện thoại", "khảo sát", "xin lại", "hậu tạ",
        "cái ô", "thất lạc",
    ],
}

_TOPIC_PATTERNS: Dict[str, List[re.Pattern]] = {
    topic: [
        re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", flags=re.IGNORECASE)
        for kw in kws
    ]
    for topic, kws in _TOPIC_KEYWORDS.items()
}


def classify_topic(text: str) -> str:
    """Phân loại chủ đề bài viết/bình luận sinh viên theo ranh giới từ chuẩn."""
    if not isinstance(text, str) or not text.strip():
        return "Giao lưu & Khác"

    normalized = clean_text(text) or text.lower()
    best_topic = "Giao lưu & Khác"
    best_score = 0
    for topic, patterns in _TOPIC_PATTERNS.items():
        score = sum(1 for p in patterns if p.search(normalized))
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic
