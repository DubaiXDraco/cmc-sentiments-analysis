# Phân tích Cảm xúc Sinh viên Trường Đại học CMC (`cmc-student-sentiment-final`)

Hệ thống phân loại cảm xúc 3 lớp (**Negative / Neutral / Positive**) và nhận diện **6 chủ đề sinh viên** trên bình luận & bài đăng Facebook cộng đồng Trường Đại học CMC (đã loại bỏ bài PR Fanpage trường và ẩn danh hóa PII), kết hợp đánh giá định lượng không rò rỉ dữ liệu, pipeline cập nhật hàng ngày và giao diện Streamlit + Plotly.

## Kết quả Thực nghiệm Chính (Không Rò Rỉ Dữ Liệu)

- **Dữ liệu thật sau ẩn danh & làm sạch:** `1868` bản ghi (`1174` comment, `694` post; `2025-11-29` → `2026-11-08`; đã loại `751` bài PR từ Fanpage trường; `0` dòng sót PII).
- **Chia tập theo nhóm `core_text` (`1860` mẫu duy nhất):** `Train / Val / Test = 1302 / 279 / 279` (`0` rò rỉ giữa các tập).
- **Kiểm định thủ công trên tập thật (`data/gold_real_test.csv`, `120` mẫu từ Test):** Tỷ lệ đồng thuận nhãn bán tự động vs kiểm định tay đạt `0.5667` (Cohen's kappa = `0.3500`), phản ánh rõ lệch miền trên tiếng lóng sinh viên.

| Mô hình | Val Acc / Macro-F1 | Semi-Auto Test Acc / F1 | Gold Real Test (120) Acc / F1 | Challenge Test (40) Acc / F1 | Độ trễ CPU | Dung lượng |
|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | `0.6093` / `0.5694` | `0.6344` / `0.5807` | `0.6083` / `0.5108` | `0.5750` / `0.5537` | `14.04 ms` | `0.30 MB` |
| **TF-IDF (Word+Char) + LinearSVC** | `0.6487` / `0.5422` | `0.6703` / `0.5214` | `0.7000` / `0.5461` | `0.3500` / `0.3333` | `49.48 ms` | `1.35 MB` |
| **Fine-tuned PhoBERT (`vinai/phobert-base-v2`)** | **`0.6918`** / **`0.6690`** | **`0.6523`** / **`0.6052`** | `0.6250` / **`0.5566`** | **`0.8750`** / **`0.8611`** | `83.33 ms` | `516.95 MB` |

- **30 lần chia lặp (`TF-IDF + LR`, `test=15%`, `seed 0..29`):** Accuracy `0.6382 ± 0.0275`, Macro-F1 `0.5979 ± 0.0323`.
- **Điểm nổi bật:** Trên bộ thử thách **40 câu phủ định / hỗn hợp / ngoài chủ đề** (`data/challenge_test_set.csv`), `Fine-tuned PhoBERT` tăng vọt từ `0.5750` (`TF-IDF + LR`) lên **`0.8750`** Accuracy (**Macro-F1 `0.8611`**, đạt `100%` Recall lớp Negative).

## Hướng dẫn Cài đặt & Chạy Tái lập (Seed = 42)

Tương thích **Python 3.10 – 3.13** (`requirements.txt` không ghim cứng phiên bản xung đột).

```bash
# 1. Cài đặt môi trường
python -m venv .venv
.venv\Scripts\activate      # Windows (hoặc source .venv/bin/activate trên Linux/macOS)
pip install -r requirements-dev.txt

# 2. (Tuỳ chọn nếu chạy lại từ đầu với HuggingFace)
python scripts/00_download_hf_models.py
python scripts/01_ingest_and_clean.py
python scripts/01b_semi_auto_label.py --seed 42

# 3. Huấn luyện & đánh giá mô hình
python scripts/02_train_baselines.py --seed 42 --n-splits 30
python scripts/02b_train_phobert.py --epochs 6 --batch-size 32 --lr 5e-5 --max-length 64 --unfreeze-layers 2 --seed 42
python scripts/02c_evaluate_all.py

# 4. Chạy pipeline hàng ngày & lập lịch tự động
python scripts/03_daily_pipeline.py --mode replay --date 2026-11-08 --backend tfidf_lr
python scripts/03b_scheduler.py --demo-interval-sec 1 --max-runs 1

# 5. Chạy giao diện demo Streamlit & bộ kiểm thử tự động
streamlit run app_demo.py
pytest -v
```

## Cấu trúc Dự án

```text
app_demo.py                         # Ứng dụng Streamlit + Plotly (4 tab, chọn động cơ PhoBERT / LR / SVM)
src/
  anonymize.py                      # Băm SHA-256 ID & xóa/che SĐT, email, IP, MSSV, URL, @mention
  preprocess.py                     # Pipeline tiền xử lý thống nhất + teencode.json + core_text chống rò rỉ
  topics.py                         # Phân loại 6 chủ đề đời sống sinh viên CMC theo ranh giới từ
  inference.py                      # SentimentEngine thống nhất + giải thích từ khóa cho mô hình tuyến tính
data/
  teencode.json                     # Từ điển viết tắt đã sửa lỗi ("cmc", "bt", "k5"...)
  cmc_student_clean.csv             # 1868 mẫu thật đã ẩn danh (không chứa JSON thô PII)
  cmc_student_labeled.csv           # 1868 mẫu kèm nhãn bán tự động và chủ đề
  train.csv / val.csv / test.csv    # 1302 / 279 / 279 (0 trùng lặp core_text)
  gold_real_test.csv                # 120 mẫu thật trích từ Test, kiểm định nhãn thủ công từng dòng
  challenge_test_set.csv            # 40 câu kiểm tra phủ định, hỗn hợp, ngoài miền
  annotation_batch_for_team.csv     # 200 mẫu chuẩn bị cho nhóm 2 người gán chéo tính Cohen's kappa
  DATA_CARD.md                      # Mô tả dữ liệu, nguồn gốc, giấy phép và giới hạn
scripts/                            # Toàn bộ script từ 00 -> 07 (ingest, train, eval, daily, schedule, slides, zip)
notebooks/01_preprocess_eda.ipynb   # Notebook EDA đã thực thi đầy đủ kèm 5 biểu đồ
reports/                            # Báo cáo chi tiết Bao_cao_toan_bo_du_an.md, JSON kết quả & biểu đồ PNG
slides/                             # Slide thuyết trình PowerPoint 12 trang
tests/                              # 6 file pytest tự động (tiền xử lý, chống rò rỉ, mô hình, pipeline, docs, app)
```

## Tài liệu Đọc thêm
- Báo cáo tổng hợp đầy đủ theo chuẩn đề cương môn học (mục 5.1–5.3 & Tuần 1–4): [reports/Bao_cao_toan_bo_du_an.md](reports/Bao_cao_toan_bo_du_an.md)
- Data Card & Quy trình đạo đức dữ liệu: [data/DATA_CARD.md](data/DATA_CARD.md)
- Quy chuẩn gán nhãn nhóm: [docs/ANNOTATION_GUIDELINE.md](docs/ANNOTATION_GUIDELINE.md)
