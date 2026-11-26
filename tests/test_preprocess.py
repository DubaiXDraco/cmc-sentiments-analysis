import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anonymize import anonymize_id, anonymize_text, contains_pii
from src.preprocess import clean_text, make_core_text, make_loose_key
from src.topics import classify_topic


def test_anonymize_pii_scrubbing():
    sample = (
        "Liên hệ 0912345678 hoặc +84987654321, email sv@cmc.edu.vn, "
        "MSSV bgd250058, IP 169.150.243.8:19035, link https://forms.gle/abc, "
        "@nguyenvana hoặc nhắn tin Nguyễn Văn A"
    )
    anon = anonymize_text(sample)
    assert not contains_pii(anon)
    assert "0912345678" not in anon
    assert "sv@cmc.edu.vn" not in anon
    assert "bgd250058" not in anon
    assert "169.150.243.8" not in anon
    assert "forms.gle" not in anon
    assert "@nguyenvana" not in anon
    assert "Nguyễn Văn A" not in anon


def test_anonymize_id_deterministic():
    a = anonymize_id("1424542162993445")
    b = anonymize_id("1424542162993445")
    c = anonymize_id("9999999999999999")
    assert a == b
    assert a != c
    assert a.startswith("cmc_") and len(a) == 16


def test_fixed_teencode_bugs():
    # Bug cũ: "cmc" -> "đại học cmc" làm "Trường Đại học CMC" thành "đại học đại học cmc"
    out_cmc = clean_text("Trường Đại học CMC tuyển sinh K5 và K4")
    assert "đại học đại học" not in out_cmc
    assert "k5" in out_cmc and "k4" in out_cmc

    # Bug cũ: "bt" -> "bình thường" làm "ko bt" thành "không bình thường"
    out_bt = clean_text("Ai bt lịch thi ck chưa, mình ko bt luôn")
    assert "bình thường" not in out_bt
    assert "ai bt lịch thi cuối kỳ chưa" in out_bt
    assert "không bt" in out_bt

    # Đảm bảo "k" đứng riêng vẫn thành "không", "ck" -> "cuối kỳ"
    assert clean_text("k thích cách dạy vs vl") == "không thích cách dạy với vãi"
    assert "chồng" not in clean_text("lịch thi ck")


def test_clean_removes_noise_and_repeated_chars():
    out = clean_text("Wifi yếu quáááá 😭 http://x.com @abc #tag !!!")
    assert out == "wifi yếu quá"


def test_empty_and_none():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""
    assert make_core_text(None) == ""
    assert make_loose_key(None) == ""


def test_core_text_ignores_prefix_suffix_intensifiers():
    base = make_core_text("Giảng viên nhiệt tình")
    variants = [
        "Thật sự Giảng viên nhiệt tình ạ",
        "Mọi người ơi, Giảng viên nhiệt tình quá !!!",
        "Giảng viên nhiệt tình lắm nha 😭",
        "Giảng viên cực kỳ nhiệt tình luôn",
    ]
    for v in variants:
        assert make_core_text(v) == base


def test_topic_classification():
    assert classify_topic("CLB Khởi nghiệp HATCH tuyển thành viên Gen 4") == "Hoạt động & CLB"
    assert classify_topic("Wifi tầng 4 phòng lab hơi chậm") == "Cơ sở vật chất & CNTT"
    assert classify_topic("Em để quên thẻ sinh viên ở phòng 403") == "Tìm đồ & Hỗ trợ"
    assert classify_topic("Tìm bạn nữ ở ghép phòng vệ sinh khép kín Vạn Phúc") == "Đời sống & Nhà trọ"
    assert classify_topic("Cho hỏi lịch thi cuối kỳ môn Toán rời rạc") == "Học vụ & Giảng viên"
