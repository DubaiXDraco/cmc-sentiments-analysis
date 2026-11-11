import os
import re
import sys
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thử import underthesea, nếu Python 3.14 chưa hỗ trợ thì dùng bộ ghép từ nội bộ
try:
    from underthesea import word_tokenize as uts_tokenize
    HAS_UTS = True
except ImportError:
    HAS_UTS = False

TEENCODE_DICT = {
    "ko": "không", "k": "không", "khum": "không", "hong": "không", "hok": "không",
    "dc": "được", "đc": "được", "dk": "đăng ký", "đk": "đăng ký", "dky": "đăng ký",
    "sv": "sinh viên", "gv": "giảng viên", "cmcu": "đại học cmc", "cmc": "đại học cmc",
    "hp": "học phần", "tc": "tín chỉ", "đrl": "điểm rèn luyện", "drl": "điểm rèn luyện",
    "ktx": "ký túc xá", "csvc": "cơ sở vật chất", "cntt": "công nghệ thông tin",
    "it": "công nghệ thông tin", "pass": "qua môn", "tạch": "trượt môn", "rớt": "trượt môn",
    "lag": "chậm", "web": "trang mạng", "app": "ứng dụng", "ad": "quản trị viên",
    "iu": "yêu", "ok": "tốt", "oke": "tốt", "đỉnh": "tuyệt vời", "xịn": "tốt",
    "chán": "tệ", "ức chế": "tệ", "bực": "tệ", "mn": "mọi người", "ae": "anh em",
    "ib": "nhắn tin", "rep": "trả lời", "cmt": "bình luận"
}

COMPOUND_WORDS = [
    "cơ sở vật chất", "công nghệ thông tin", "điểm rèn luyện", "đại học cmc",
    "sinh viên", "giảng viên", "đăng ký", "tín chỉ", "học phần", "học phí",
    "học bổng", "ký túc xá", "thư viện", "điều hòa", "thang máy", "nhà xe",
    "tuyệt vời", "nhiệt tình", "hiện đại", "hỗ trợ", "qua môn", "trượt môn",
    "cảm ơn", "mọi người", "anh em", "nhắn tin", "trả lời", "bình luận",
    "quản trị viên", "trang mạng", "ứng dụng", "cuối kỳ", "bài tập", "tài liệu"
]

def segment_vietnamese(text):
    if HAS_UTS:
        return uts_tokenize(text, format="text")
    for cw in COMPOUND_WORDS:
        text = re.sub(r"\b" + cw + r"\b", cw.replace(" ", "_"), text)
    return text

def clean_vietnamese_text(text, segment_for_phobert=True):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"http\S+|www\S+|https\S+|\S+@\S+|\b\d{9,11}\b", " ", text)
    text = re.sub(r"[❤️🥰😍👍🔥💯]", " tuyệt vời ", text)
    text = re.sub(r"[😡🤬😭😢👎💔]", " tệ buồn ", text)
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    words = [TEENCODE_DICT.get(w, w) for w in text.split()]
    text = re.sub(r"\s+", " ", " ".join(words)).strip()
    if segment_for_phobert and len(text) > 2:
        text = segment_vietnamese(text)
    return text

def auto_label_sentiment(text):
    pos_words = ["tuyệt_vời", "tốt", "đẹp", "nhiệt_tình", "hiện_đại", "thích", "yêu", "hỗ_trợ", "nhanh", "xịn", "vui", "cảm_ơn", "qua_môn", "học_bổng", "mát"]
    neg_words = ["tệ", "chậm", "lỗi", "trượt_môn", "đắt", "nóng", "hỏng", "bực", "chán", "kém", "tắc", "thiếu", "ồn", "mất", "lâu", "đông", "buồn"]
    score = sum(1 for pw in pos_words if pw in text) - sum(1 for nw in neg_words if nw in text)
    return "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral")

def generate_mock_cmc_data(n_samples=3500):
    templates = [
        ("Cơ sở vật chất ở Vạn Phúc xịn thật sự, điều hòa mát rượi ❤️", "2026-10-01"),
        ("Web đk tín chỉ hôm nay lag quá ad ơi, quay mòng mòng ko vào đc 😭", "2026-10-02"),
        ("Cho mình hỏi lịch thi cuối kỳ môn Cấu trúc dữ liệu xem ở đâu vậy mn?", "2026-10-03"),
        ("Gv khoa CNTT dạy siêu nhiệt tình, hỏi bài lúc nửa đêm vẫn rep", "2026-10-04"),
        ("Thang máy giờ cao điểm chờ hơi lâu, mong trường mở thêm luồng đi", "2026-10-05"),
        ("Học bổng năm nay xét duyệt nhanh gọn, cảm ơn phòng Đào tạo!", "2026-10-06"),
        ("Có ai lập nhóm làm bài tập lớn môn Khoa học dữ liệu khum cho mình ké với", "2026-10-07"),
        ("Wifi tầng 4 hôm nay hơi chậm, k load nổi tài liệu trên LMS", "2026-10-08"),
        ("Sự kiện chào tân sv năm nay cháy quá, âm thanh ánh sáng đỉnh 💯", "2026-10-09"),
        ("Thủ tục xin giấy xác nhận sv làm ở phòng nào thế ae?", "2026-10-10"),
        ("Nhà xe hôm nay đông quá, xếp hàng mãi mới lấy đc xe về", "2026-10-11"),
        ("Thư viện yên tĩnh, nhiều ổ cắm điện, ngồi code cả ngày cực thích 👍", "2026-10-12")
    ]
    np.random.seed(42)
    rows = []
    for i in range(n_samples):
        tpl, _ = templates[i % len(templates)]
        day = np.random.randint(1, 29)
        rows.append({"id": i + 1, "raw_text": tpl, "timestamp": f"2026-10-{day:02d}"})
    return pd.DataFrame(rows)

def run_pipeline(raw_csv_path="data/raw/cmc_fb_raw.csv", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    if os.path.exists(raw_csv_path):
        df = pd.read_csv(raw_csv_path)
    else:
        df = generate_mock_cmc_data(3500)
        df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
        
    df.dropna(subset=["raw_text"], inplace=True)
    df["clean_text"] = df["raw_text"].apply(lambda x: clean_vietnamese_text(x, segment_for_phobert=True))
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 2].copy()
    
    df["sentiment"] = df["clean_text"].apply(auto_label_sentiment)
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df["label"] = df["sentiment"].map(label_map)
    
    clean_path = os.path.join(output_dir, "cmc_clean_labeled.csv")
    df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] Hoàn tất! Tổng: {len(df)} mẫu | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    run_pipeline()
