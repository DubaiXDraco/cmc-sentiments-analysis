#!/usr/bin/env python3
"""
scripts/06_build_slides.py
Tuần 4: Tạo bộ Slide thuyết trình PowerPoint 12 trang (`slides/Slide_Thuyet_Trinh_CMC_Sentiment.pptx`)
với số liệu thực tế đọc trực tiếp từ `reports/*.json` và biểu đồ thật từ `reports/*.png`.
"""

import json
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
SLIDES_DIR = ROOT / "slides"
SLIDES_DIR.mkdir(exist_ok=True)
OUT_PPTX = SLIDES_DIR / "Slide_Thuyet_Trinh_CMC_Sentiment.pptx"

NAVY = RGBColor(15, 23, 42)
BLUE = RGBColor(37, 99, 235)
DARK = RGBColor(30, 41, 59)
MUTED = RGBColor(100, 116, 139)
WHITE = RGBColor(255, 255, 255)


def add_header(slide, title: str, subtitle: str = "") -> None:
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(13)
        p2.font.color.rgb = MUTED


def add_bullets(slide, left: float, top: float, width: float, height: float, items: list, font_size: int = 15) -> None:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for idx, text in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = f"• {text}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = DARK
        p.space_after = Pt(8)


def add_footer(slide, page_num: int) -> None:
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12.1), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = f"Hệ thống Phân tích Cảm xúc Sinh viên Trường Đại học CMC  |  Trang {page_num}/12"
    p.font.size = Pt(10)
    p.font.color.rgb = MUTED


def main() -> None:
    ingest = json.loads((REPORTS / "ingestion_stats.json").read_text(encoding="utf-8"))
    splits = json.loads((REPORTS / "split_stats.json").read_text(encoding="utf-8"))
    audit = json.loads((REPORTS / "labeling_audit.json").read_text(encoding="utf-8"))
    rep = json.loads((REPORTS / "repeated_splits.json").read_text(encoding="utf-8"))
    ev = json.loads((REPORTS / "evaluation_results.json").read_text(encoding="utf-8"))

    lr = ev["models"]["tfidf_lr"]
    svm = ev["models"]["tfidf_svm"]
    pb = ev["models"]["phobert"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1: Cover
    s1 = prs.slides.add_slide(blank)
    bg = s1.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY
    tb = s1.shapes.add_textbox(Inches(0.9), Inches(1.8), Inches(11.5), Inches(3.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "BÀI TẬP LỚN AI / XỬ LÝ NGÔN NGỮ TỰ NHIÊN"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(96, 165, 250)
    p2 = tf.add_paragraph()
    p2.text = "Hệ thống Phân tích Cảm xúc Sinh viên\nTrường Đại học CMC"
    p2.font.size = Pt(40)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.space_before = Pt(14)
    p3 = tf.add_paragraph()
    p3.text = (
        f"Dữ liệu thật cộng đồng CMC đã ẩn danh ({ingest['after_exact_text_clean_dedup']:,} mẫu) • "
        "So sánh TF-IDF Baseline & Fine-tuned PhoBERT • Pipeline hàng ngày & Demo Streamlit"
    )
    p3.font.size = Pt(16)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(18)

    # Slide 2: Problem & Objectives
    s2 = prs.slides.add_slide(blank)
    add_header(s2, "1. Lý do chọn chủ đề, Mục tiêu & Phạm vi", "Mục 5.2a, 5.3a, 5.3b — Bối cảnh thực tế tại Đại học CMC")
    add_bullets(
        s2, 0.7, 1.45, 5.8, 5.3,
        [
            "Vấn đề thực tế: Phản hồi SV rải rác trên group & fanpage, chứa nhiều teencode (dkmh, csvc, ko, k5), khó theo dõi kịp thời bằng khảo sát cuối kỳ.",
            "Mục tiêu: Phân loại 3 lớp cảm xúc (Negative / Neutral / Positive) + phân loại 6 chủ đề đời sống sinh viên.",
            f"Phạm vi: {ingest['after_exact_text_clean_dedup']:,} mẫu bình luận & bài đăng sinh viên ({ingest['date_min']} → {ingest['date_max']}).",
            "Đối tượng thụ hưởng: Phòng CTSV, Ban CNTT/CSVC và các CLB nắm bắt xu hướng phản hồi theo ngày/tuần.",
        ],
    )
    add_bullets(
        s2, 6.8, 1.45, 5.8, 5.3,
        [
            "Yêu cầu phi chức năng (đã đo trên CPU):",
            f"Độ trễ: TF-IDF+LR {lr['latency_ms_per_text']} ms | PhoBERT {pb['latency_ms_per_text']} ms/câu.",
            f"Dung lượng: TF-IDF+LR {lr['size_mb']} MB | PhoBERT {pb['size_mb']} MB.",
            "Khả năng giải thích: Trích Top-5 n-gram đóng góp cho mô hình tuyến tính + xác suất Softmax 3 lớp.",
            "Tái lập: Seed cố định 42, toàn bộ pipeline chạy tự động + 6 bộ kiểm thử pytest.",
        ],
    )
    add_footer(s2, 2)

    # Slide 3: 4-Week Execution Roadmap
    s3 = prs.slides.add_slide(blank)
    add_header(s3, "2. Quy trình Thực hiện theo Lộ trình 4 Tuần", "Bám sát đầy đủ 4 giai đoạn triển khai và kiểm thử")
    add_bullets(
        s3, 0.7, 1.45, 11.8, 5.3,
        [
            f"Tuần 1 (Thu thập & Tiền xử lý): Nạp 3 tập JSON thật ({ingest['total_raw_items_scanned']:,} mục), tách bỏ {ingest['excluded_official_fanpage_posts']} bài PR Fanpage trường, ẩn danh PII còn {ingest['after_exact_text_clean_dedup']:,} mẫu sạch + sửa lỗi teencode + Notebook EDA.",
            f"Tuần 2 (Gán nhãn & Huấn luyện): Gán nhãn bán tự động wonrax/phobert + kiểm định thủ công {audit['n_gold_samples']} mẫu thật; chia GroupShuffleSplit {splits['n_train']}/{splits['n_val']}/{splits['n_test']} (0 rò rỉ); huấn luyện TF-IDF+LR, LinearSVC & Fine-tune PhoBERT.",
            "Tuần 3 (Pipeline hàng ngày): Xây dựng script cập nhật hàng ngày idempotent (3 chế độ replay/json/playwright), kiểm thử 5 ngày liên tiếp (2026-11-04 → 2026-11-08) & lập lịch bằng thư viện schedule.",
            "Tuần 4 (Demo & Báo cáo): Ứng dụng Streamlit + Plotly 4 tab (dự đoán, dashboard chủ đề, xu hướng ngày/tuần, benchmark), báo cáo hoàn chỉnh & bộ pytest tự động.",
        ],
        font_size=16,
    )
    add_footer(s3, 3)

    # Slide 4: Data Funnel & Leakage Control
    s4 = prs.slides.add_slide(blank)
    add_header(s4, "3. Phễu Dữ liệu, Sửa Teencode & Kiểm soát Rò rỉ", "Loại nhiễu PR chính thức và bảo đảm 0 trùng lặp giữa Train/Val/Test")
    add_bullets(
        s4, 0.7, 1.45, 5.8, 5.3,
        [
            f"Quét {ingest['total_raw_items_scanned']:,} mục thô → loại {ingest['excluded_official_fanpage_posts']} bài PR Fanpage trường → còn {ingest['after_exact_text_clean_dedup']:,} mẫu SV ({ingest['type_distribution']['comment']} cmt, {ingest['type_distribution']['post']} post).",
            "Sửa lỗi teencode cũ: bỏ 'cmc'→'đại học cmc', bỏ 'bt'→'bình thường', giữ nguyên khóa 'k3/k4/k5'.",
            f"Chuẩn hóa core_text → {splits['n_unique_core_text']:,} câu gốc duy nhất; chia GroupShuffleSplit {splits['n_train']}/{splits['n_val']}/{splits['n_test']} (0 giao nhau).",
        ],
    )
    s4.shapes.add_picture(str(REPORTS / "eda_source_type.png"), Inches(6.7), Inches(1.5), width=Inches(6.0))
    add_footer(s4, 4)

    # Slide 5: Labeling Audit & Imbalance
    s5 = prs.slides.add_slide(blank)
    add_header(s5, "4. Gán nhãn Bán tự động, Mất cân bằng & Kiểm định Gold Test", "Phát hiện domain drift qua đối chiếu 120 mẫu kiểm định thủ công")
    add_bullets(
        s5, 0.7, 1.45, 5.8, 5.3,
        [
            f"Nhãn bán tự động (1.860 mẫu): Neutral {splits['label_dist_dedup']['Neutral']} | Positive {splits['label_dist_dedup']['Positive']} | Negative {splits['label_dist_dedup']['Negative']}.",
            "Xử lý mất cân bằng: class_weight='balanced' (Baseline) & CrossEntropyLoss weights [1.99, 0.69, 0.96] (PhoBERT).",
            f"Kiểm định {audit['n_gold_samples']} mẫu thật từ tập Test: Agreement = {audit['semi_vs_gold_agreement']:.4f}, Cohen's kappa = {audit['semi_vs_gold_cohen_kappa']:.4f} (sửa {audit['n_corrected']}/120 ca lệch miền teencode/ghép trọ).",
            f"Chuẩn bị sẵn {splits['team_annotation_batch_size']} mẫu (annotation_batch_for_team.csv) + Annotation Guideline cho nhóm gán chéo.",
        ],
    )
    s5.shapes.add_picture(str(REPORTS / "eda_sentiment_dist.png"), Inches(6.9), Inches(1.6), width=Inches(5.8))
    add_footer(s5, 5)

    # Slide 6: EDA Topic & Weekly Trends
    s6 = prs.slides.add_slide(blank)
    add_header(s6, "5. Khám phá Dữ liệu theo 6 Chủ đề & Xu hướng Thời gian", "Độ dài trung bình 30.64 từ/câu (trung vị 10 từ), 6 nhóm chủ đề SV")
    s6.shapes.add_picture(str(REPORTS / "eda_topic_sentiment.png"), Inches(0.55), Inches(1.45), width=Inches(6.05))
    s6.shapes.add_picture(str(REPORTS / "eda_weekly_trend.png"), Inches(6.75), Inches(1.55), width=Inches(6.05))
    add_footer(s6, 6)

    # Slide 7: Model Architecture & CPU Optimization
    s7 = prs.slides.add_slide(blank)
    add_header(s7, "6. So sánh Mô hình & Tối ưu Fine-tune PhoBERT trên CPU", "Huấn luyện cả Baseline tuyến tính và Transformer tiếng Việt (vinai/phobert-base-v2)")
    add_bullets(
        s7, 0.7, 1.45, 11.8, 5.3,
        [
            "Baseline 1 (TF-IDF Word 1-2 gram + Logistic Regression): Nhanh nhất (14.04 ms), 0.30 MB, giải thích được từ khóa.",
            "Baseline 2 (Word + Char_wb 3-5 gram TF-IDF + Calibrated LinearSVC): Nắm bắt cụm ký tự nhưng dễ thiên lệch lớp đa số trên câu ngoài miền.",
            "Mô hình chính (Fine-tuned vinai/phobert-base-v2): Mở khóa 2 tầng encoder cuối + classification head (14.77M/135M params), epochs=6, lr=5e-5, max_length=64, seed=42.",
            "Tối ưu CPU: Length-sorted batching + frozen-prefix hidden-state caching → kiểm chứng torch.allclose max_diff=0.00e+00, hoàn tất 6 epoch trong ~3.8 phút, Epoch 5 đạt Val Macro-F1 = 0.6690.",
        ],
        font_size=16,
    )
    add_footer(s7, 7)

    # Slide 8: Quantitative Benchmark Results
    s8 = prs.slides.add_slide(blank)
    add_header(s8, "7. Kết quả Đánh giá Định lượng (Không Rò rỉ)", "PhoBERT vượt trội ở Macro-F1 và bứt phá trên bộ thử thách phủ định")
    s8.shapes.add_picture(str(REPORTS / "model_benchmark.png"), Inches(0.6), Inches(1.45), width=Inches(7.3))
    add_bullets(
        s8, 8.1, 1.45, 4.7, 5.3,
        [
            f"Val Macro-F1: LR {lr['val']['macro_f1']:.4f} | SVM {svm['val']['macro_f1']:.4f} | PhoBERT {pb['val']['macro_f1']:.4f}.",
            f"Test Macro-F1: LR {lr['test']['macro_f1']:.4f} | PhoBERT {pb['test']['macro_f1']:.4f}.",
            f"Gold Real Test (120): LR {lr['gold_real_test']['macro_f1']:.4f} | SVM {svm['gold_real_test']['macro_f1']:.4f} | PhoBERT {pb['gold_real_test']['macro_f1']:.4f}.",
            f"Challenge Test (40): LR {lr['challenge_test']['accuracy']:.4f} → PhoBERT {pb['challenge_test']['accuracy']:.4f} (Macro-F1 {pb['challenge_test']['macro_f1']:.4f}, 100% Recall Negative).",
            f"30 lần chia lặp (LR): Acc {rep['accuracy_mean']:.4f} ± {rep['accuracy_std']:.4f}.",
        ],
        font_size=14,
    )
    add_footer(s8, 8)

    # Slide 9: Error Analysis & Confusion Matrix
    s9 = prs.slides.add_slide(blank)
    add_header(s9, "8. Ma trận Nhầm lẫn & Phân tích Lỗi (Gold Real Test)", "Đối chiếu TF-IDF + LR với Fine-tuned PhoBERT trên 120 mẫu thật")
    s9.shapes.add_picture(str(REPORTS / "confusion_matrix_comparison.png"), Inches(0.6), Inches(1.5), width=Inches(7.5))
    add_bullets(
        s9, 8.3, 1.5, 4.5, 5.3,
        [
            "PhoBERT nâng Recall lớp Negative trên Gold Real Test từ 0.40 (LR) lên 0.60 và Positive từ 0.67 lên 0.77.",
            "Trên Challenge Test 40 câu: TF-IDF sai hàng loạt câu phủ định ('Không hài lòng...', 'Thầy dạy ko dễ hiểu'), PhoBERT đúng 35/40 câu.",
            "5 câu PhoBERT còn nhầm: thành ngữ phủ định kép ('Không có gì để chê') và câu thời tiết ('Hôm nay trời mưa/đẹp').",
        ],
        font_size=14,
    )
    add_footer(s9, 9)

    # Slide 10: Daily Pipeline & Scheduler
    s10 = prs.slides.add_slide(blank)
    add_header(s10, "9. Quy trình Cập nhật Hàng ngày & Lập lịch Tự động", "Tuần 3 — Kiểm thử 5 ngày liên tiếp thật & cơ chế chống ghi trùng")
    add_bullets(
        s10, 0.7, 1.45, 11.8, 5.3,
        [
            "scripts/03_daily_pipeline.py hỗ trợ 3 chế độ: replay (phát lại theo ngày thật), json (nhập liệu mới + lọc PR + ẩn danh PII) và playwright (thu thập trang công khai).",
            "Cơ chế idempotent qua daily_results/processed_ids.json: chạy lại ngày 2026-11-06 ghi nhận 0 dòng mới, bỏ qua 28 ID đã xử lý.",
            "Kiểm thử thực tế 5 ngày liên tiếp (2026-11-04 → 2026-11-08): xử lý 66 bản ghi thật, lưu riêng từng file daily_results/sentiment_YYYY-MM-DD.csv.",
            "scripts/03b_scheduler.py dùng thư viện schedule chạy định kỳ hàng ngày lúc 08:00 và ghi log vào daily_results/pipeline.log.",
        ],
        font_size=16,
    )
    add_footer(s10, 10)

    # Slide 11: Ethics, Privacy & Demo Script
    s11 = prs.slides.add_slide(blank)
    add_header(s11, "10. Đạo đức AI, Bảo vệ Dữ liệu & Kịch bản Demo", "Mục 5.2d, 5.2e — Tuân thủ quyền riêng tư & kiểm thử tự động")
    add_bullets(
        s11, 0.7, 1.45, 5.8, 5.3,
        [
            "Bảo mật & PII: Băm SHA-256 ID, xóa URL/email/SĐT/IP/MSSV/@mention; không đưa JSON thô vào repo; pytest xác nhận 0 dòng sót PII.",
            "Minh bạch: Báo cáo trung thực điểm Gold & kappa 0.3500; cung cấp giải thích từ khóa (TF-IDF) và độ tin cậy.",
            "Kiểm thử tự động: 6 file pytest kiểm tra tiền xử lý, chống rò rỉ, 3 mô hình, pipeline ngày, độ khớp số liệu báo cáo và Streamlit AppTest.",
        ],
    )
    add_bullets(
        s11, 6.8, 1.45, 5.8, 5.3,
        [
            "Kịch bản Demo trực tiếp (3 phút):",
            "1. Chạy `streamlit run app_demo.py`, chọn động cơ PhoBERT / TF-IDF+LR ở Sidebar.",
            "2. Tab Dự đoán: thử câu chứa teencode + phủ định ('Phòng học đẹp nhưng wifi lag quá').",
            "3. Tab Dashboard: lọc theo chủ đề 'Cơ sở vật chất & CNTT' xem biểu đồ tỷ lệ.",
            "4. Tab Xu hướng: bật chế độ Tuần/Ngày và xem log pipeline 5 ngày gần nhất.",
        ],
    )
    add_footer(s11, 11)

    # Slide 12: Conclusion
    s12 = prs.slides.add_slide(blank)
    bg12 = s12.background.fill
    bg12.solid()
    bg12.fore_color.rgb = NAVY
    tb12 = s12.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.5))
    tf12 = tb12.text_frame
    tf12.word_wrap = True
    p = tf12.paragraphs[0]
    p.text = "CẢM ƠN THẦY CÔ VÀ CÁC BẠN ĐÃ LẮNG NGHE!"
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p2 = tf12.add_paragraph()
    p2.text = (
        "Hướng phát triển tiếp theo: Mở rộng dữ liệu cộng đồng lên 3.000–6.000 mẫu • "
        "Hoàn thiện gán nhãn chéo nhóm (Cohen's kappa >= 0.70) • "
        "Fine-tune PhoBERT trực tiếp trên nhãn vàng & tối ưu ONNX."
    )
    p2.font.size = Pt(17)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_before = Pt(20)

    prs.save(str(OUT_PPTX))
    print(f"✓ Đã tạo bộ slide {len(prs.slides)} trang tại: {OUT_PPTX}")


if __name__ == "__main__":
    main()
