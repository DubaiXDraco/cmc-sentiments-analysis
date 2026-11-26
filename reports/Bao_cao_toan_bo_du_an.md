# Báo cáo Toàn bộ Dự án Bài tập lớn: Hệ thống Phân tích Cảm xúc Sinh viên Trường Đại học CMC

**Hình thức thực hiện (Mục 5.1):** Bài tập lớn nhóm 3–5 sinh viên, tiếp cận bài toán thực tiễn bằng kỹ thuật Học máy (ML) và Xử lý ngôn ngữ tự nhiên (NLP) tiếng Việt, kết hợp pipeline vận hành thử nghiệm hàng ngày và giao diện trực quan hóa dữ liệu.

---

## 1. Lý do chọn chủ đề & Tính thực tiễn (Mục 5.2a, 5.3a)

### 1.1 Bối cảnh và vấn đề
Trong môi trường đại học, phản hồi hàng ngày của sinh viên về cơ sở vật chất, cổng đăng ký môn học, học phí, giảng viên và hoạt động câu lạc bộ xuất hiện liên tục trên các hội nhóm và fanpage cộng đồng (như *Khu Tự Trị CMC*, *Cộng Đồng Sinh Viên CMC*, fanpage *TruongDaiHocCMC* và *ctsv.cmcuni*). Tuy nhiên, các thông tin này nằm rải rác, chứa nhiều từ viết tắt sinh viên (`teencode`), câu hỏi đan xen cảm xúc và chưa được tổng hợp theo chủ đề hay theo dõi diễn biến theo ngày/tuần.

### 1.2 Đối tượng sử dụng & Lợi ích so với cách làm truyền thống
- **Đối tượng thụ hưởng:** Phòng Công tác Sinh viên (CTSV), Ban Quản trị CNTT/CSVC và Ban chấp hành Đoàn – Hội/CLB Trường Đại học CMC.
- **So sánh với khảo sát truyền thống (Google Form định kỳ cuối kỳ):**

| Tiêu chí | Khảo sát truyền thống (Form/Phiếu cuối kỳ) | Hệ thống AI phân tích cảm xúc cộng đồng (Giải pháp đề xuất) |
|---|---|---|
| **Tần suất phản hồi** | Theo kỳ/năm (độ trễ hàng tháng) | Theo **ngày / tuần** qua pipeline tự động |
| **Tính tự nhiên** | Dễ bị thiên lệch do trả lời chiếu lệ | Phản ánh tâm tư tự phát trong ngữ cảnh thực tế |
| **Phân loại chủ đề** | Phải đọc thủ công từng câu góp ý mở | Tự động gắn 6 chủ đề (`CSVC & CNTT`, `Học vụ`, `CLB`, `Nhà trọ`...) |
| **Chi phí nhân lực** | Tốn thời gian tổng hợp thủ công | Tự động hóa tiền xử lý, suy luận và vẽ biểu đồ xu hướng |

---

## 2. Mục tiêu, Phạm vi & Yêu cầu Phi chức năng (Mục 5.3b)

- **Mục tiêu mô hình:** Phân loại văn bản tiếng Việt thành 3 lớp cảm xúc (**Negative**, **Neutral**, **Positive**) đồng thời phân nhóm theo **6 chủ đề đời sống sinh viên CMC**.
- **Phạm vi dữ liệu & miền áp dụng:** Bình luận và bài viết công khai của sinh viên CMC (thu thập từ 2 group cộng đồng sinh viên và bình luận dưới bài đăng của 2 fanpage trường, giai đoạn `2025-11-29` → `2026-11-08`). Không áp dụng cho mục đích chấm điểm cá nhân hay kết luận chính thức về từng cá nhân.
- **Các ràng buộc & Yêu cầu phi chức năng (đã đo thực tế trên CPU 12 luồng, RAM thông thường):**

| Yêu cầu phi chức năng | Baseline (`TF-IDF + LR`) | `TF-IDF (Word+Char) + SVM` | Mô hình chính (`PhoBERT v2`) |
|---|---|---|---|
| **Độ chính xác tổng quát (Gold Macro-F1)** | `0.5108` | `0.5461` | **`0.5566`** |
| **Xử lý phủ định / hỗn hợp (Challenge Acc)** | `0.5750` | `0.3500` | **`0.8750`** (Macro-F1 `0.8611`) |
| **Độ trễ suy luận CPU thực tế** | **`14.04 ms/câu`** | `49.48 ms/câu` | `83.33 ms/câu` (~12 câu/giây) |
| **Dung lượng mô hình trên đĩa** | **`0.30 MB`** | `1.35 MB` | `516.95 MB` |
| **Khả năng giải thích (Explainability)** | Trích xuất Top-5 n-gram đóng góp (`TF-IDF × coef_`) | Trích xuất trọng số tuyến tính trung bình | Phân bố xác suất Softmax 3 lớp + độ tin cậy |
| **Khả năng mở rộng & Tái lập** | Rất cao (pipeline Python chuẩn, `seed=42`, không phụ thuộc GPU) | Cao | Trung bình – cao (hỗ trợ cả CPU & GPU, checkpoint chuẩn HuggingFace) |

---

## 3. Dữ liệu, Làm sạch, Gán nhãn & Phân tích Khám phá (Mục 5.2b, 5.3c – Tuần 1 & 2)

### 3.1 Nguồn dữ liệu, Giấy phép & Lọc nhiễu truyền thông
- **Nguồn:** 3 tập xuất JSON công khai từ cộng đồng CMC (`2763` mục thô gồm cả bài đăng và bình luận đệ quy).
- **Giấy phép:** Dữ liệu phục vụ mục đích **nghiên cứu học thuật và bài tập lớn nội bộ**, không thương mại hóa và không tái phân phối dữ liệu thô chứa định danh.
- **Loại bỏ nhiễu PR chính thức:** Khác với cách trộn toàn bộ bài đăng Fanpage trường (làm sai lệch cảm xúc theo hướng quảng bá), `scripts/01_ingest_and_clean.py` tách bỏ **751 bài đăng (`post`) thông báo/PR** của Fanpage `TruongDaiHocCMC` và `ctsv.cmcuni`, chỉ giữ lại bài viết trong 2 group sinh viên và **toàn bộ bình luận của sinh viên** trên cả 4 nguồn.

### 3.2 Quy trình Tiền xử lý & Khắc phục lỗi Teencode
Module dùng chung `src/preprocess.py` và `data/teencode.json` bảo đảm tính nhất quán giữa lúc huấn luyện, pipeline hàng ngày và giao diện demo:
1. Chuẩn hóa Unicode `NFC` và ẩn danh hóa PII trước khi làm sạch.
2. Đưa về chữ thường, rút gọn ký tự lặp kéo dài (`quáaaaa` → `quá`), bỏ `#hashtag`, emoji và ký tự đặc biệt.
3. Thay thế `teencode` theo ranh giới từ (`(?<!\w)...(?!\w)`) và **sửa triệt để các lỗi từ điển cũ**:
   - Gỡ bỏ quy tắc gây hỏng dữ liệu `"cmc": "đại học cmc"` (tránh biến `"Trường Đại học CMC"` thành `"đại học đại học cmc"`).
   - Gỡ bỏ `"bt": "bình thường"` (tránh biến `"ko bt"` thành `"không bình thường"`).
   - Bảo vệ từ khóa `"k"` bằng lookahead `(?<!\w)k(?!\d|\w)` để không làm biến dạng tên khóa học `k3, k4, k5`.

### 3.3 Phễu dữ liệu & Kiểm soát rò rỉ (`core_text` + `GroupShuffleSplit`)
| Bước xử lý | Số mẫu | Ghi chú |
|---|---|---|
| Tổng mục thô quét từ 3 file JSON | 2763 | Gồm bài viết và bình luận lồng nhau |
| Loại bài PR chính thức Fanpage trường | -751 | Chỉ giữ tiếng nói sinh viên |
| Loại bản ghi trống / quá ngắn (`< 5` ký tự hoặc `< 2` từ) | -76 | Sau ẩn danh hóa và làm sạch |
| Sau loại trùng `text_clean` (`data/cmc_student_clean.csv`) | **1868** | `1174` comment (62.8%) / `694` post (37.2%) |
| Sau loại trùng câu gốc `core_text` | **1860** | Loại bỏ 8 biến thể chỉ khác tiền/hậu tố/từ nhấn mạnh |
| **Chia tập không rò rỉ (`seed=42`)** | **1302 / 279 / 279** | **Train / Val / Test (70/15/15)** — `0` trùng `core_text` |

### 3.4 Gán nhãn Bán tự động, Xử lý Mất cân bằng & Kiểm định Thủ công (`gold_real_test.csv`)
- **Gán nhãn bán tự động (`scripts/01b_semi_auto_label.py`):** Suy luận bằng mô hình pretrained `wonrax/phobert-base-vietnamese-sentiment` (`1837` mẫu) kết hợp quy tắc minh bạch cho tin tiện ích tìm đồ/nhà trọ không mang phàn nàn (`31` mẫu → `Neutral`).
- **Xử lý mất cân bằng lớp:** Trong 1860 mẫu duy nhất có `Neutral: 915` (49.2%), `Positive: 636` (34.2%), `Negative: 309` (16.6%). Đã xử lý bằng `class_weight="balanced"` cho Baseline và trọng số nghịch đảo tần suất `[1.9908, 0.6878, 0.9581]` trong `CrossEntropyLoss` của PhoBERT.
- **Kiểm định thủ công tập thật (loại bỏ ảo giác pseudo-label):** Trích mẫu phân tầng **120 bản ghi thật** hoàn toàn từ `data/test.csv` (40 mẫu mỗi lớp bán tự động, 0 giao với Train/Val) và rà soát gán tay từng câu thành `data/gold_real_test.csv`:
  - Phân bố nhãn kiểm định thật: `Neutral: 80 / Positive: 30 / Negative: 10`.
  - Độ đồng thuận nhãn bán tự động vs kiểm định tay: **`0.5667`** (đã hiệu chỉnh 52/120 câu), **Cohen's kappa = `0.3500`**. Kết quả chỉ ra rõ hiện tượng lệch miền của mô hình pretrained tổng quát khi gặp tiếng lóng Gen-Z, tin tìm roommate và câu trêu đùa bạn bè — qua đó chứng minh tầm quan trọng của file `data/annotation_batch_for_team.csv` (200 mẫu) và `docs/ANNOTATION_GUIDELINE.md` + `scripts/01c_cohen_kappa.py` khi mở rộng gán nhãn nhóm 3–5 sinh viên.

### 3.5 Kết quả Phân tích Khám phá (EDA – `notebooks/01_preprocess_eda.ipynb`)
- **Nguồn:** *Khu Tự Trị CMC* (1146), *Cộng Đồng Sinh Viên CMC* (422), bình luận *ctsv.cmcuni* (213), bình luận *TruongDaiHocCMC* (87).
- **Chủ đề:** *Giao lưu & Khác* (1301), *Hoạt động & CLB* (207), *Học vụ & Giảng viên* (139), *Đời sống & Nhà trọ* (103), *Cơ sở vật chất & CNTT* (71), *Tìm đồ & Hỗ trợ* (47).
- **Độ dài văn bản sạch:** Trung bình **30.64 từ/câu** (trung vị **10.0 từ**, bách phân vị thứ 75 là **20.0 từ**, tối đa 1421 từ), phản ánh đặc trưng bình luận mạng xã hội ngắn gọn nhưng vẫn có bài tâm sự dài.
- **Biểu đồ trực quan:** Đã xuất tại `reports/eda_sentiment_dist.png`, `reports/eda_topic_sentiment.png`, `reports/eda_source_type.png`, `reports/eda_length_dist.png` và `reports/eda_weekly_trend.png`.

---

## 4. Huấn luyện & Đánh giá Mô hình (Tuần 2)

### 4.1 Thiết kế Mô hình & Tối ưu Huấn luyện trên CPU
1. **Baseline 1 (`scripts/02_train_baselines.py`):** TF-IDF (word 1–2 gram, `max_features=5000`, `min_df=2`, `sublinear_tf=True`) + Logistic Regression (`class_weight="balanced"`, `seed=42`).
2. **Baseline 2 (`scripts/02_train_baselines.py`):** FeatureUnion (word 1–2 gram + char_wb 3–5 gram) + Calibrated LinearSVC (`cv=3`, `class_weight="balanced"`, `seed=42`).
3. **Mô hình chính (`scripts/02b_train_phobert.py`):** Fine-tune **`vinai/phobert-base-v2`** (mở khóa 2 tầng encoder trên cùng + classification head, `14,768,643 / 135,000,579` tham số huấn luyện, `epochs=6`, `batch_size=32`, `lr=5e-5`, `max_length=64`, `seed=42`).
   - **Tối ưu CPU:** Áp dụng *length-sorted batching + frozen-prefix hidden-state caching* giúp rút ngắn thời gian fine-tune trên CPU xuống còn **226.59 giây (~3.8 phút)** và kiểm chứng sai số giữa cache với full forward đạt `max_diff = 0.00e+00 <= 1e-4`. Epoch 5 đạt **Val Macro-F1 tốt nhất = `0.6690`**.

### 4.2 Bảng So sánh Hiệu năng Thực tế

| Mô hình | Val Acc / F1 | Semi-Auto Test Acc / F1 | Gold Real Test (120) Acc / F1 | Challenge Test (40) Acc / F1 | Độ trễ CPU | Dung lượng |
|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | `0.6093` / `0.5694` | `0.6344` / `0.5807` | `0.6083` / `0.5108` | `0.5750` / `0.5537` | `14.04 ms` | `0.30 MB` |
| **TF-IDF (Word+Char) + LinearSVC** | `0.6487` / `0.5422` | `0.6703` / `0.5214` | `0.7000` / `0.5461` | `0.3500` / `0.3333` | `49.48 ms` | `1.35 MB` |
| **Fine-tuned PhoBERT (v2)** | **`0.6918` / `0.6690`** | **`0.6523` / `0.6052`** | `0.6250` / **`0.5566`** | **`0.8750` / `0.8611`** | `83.33 ms` | `516.95 MB` |

- **Đánh giá lặp 30 lần chia (`TF-IDF + LR`, `test=15%`, `seed 0..29`):** Accuracy **`0.6382 ± 0.0275`** (min `0.5699`, max `0.6882`), Macro-F1 **`0.5979 ± 0.0323`**.
- **Nhận xét quan trọng:**
  - `LinearSVC` đạt Accuracy tổng `0.7000` trên Gold Real Test do thiên về lớp đa số `Neutral` (Recall Negative rất thấp), nên tụt xuống `0.3500` trên tập Challenge.
  - **`Fine-tuned PhoBERT`** đạt Macro-F1 cao nhất trên **cả 4 tập đánh giá** và đặc biệt vượt trội trên **Challenge Test (`0.8750` Accuracy / `0.8611` Macro-F1)** — xử lý đúng **18/18 (100% Recall)** câu phủ định/phàn nàn phức tạp như *"Không hài lòng với cách tổ chức"*, *"Thầy dạy ko dễ hiểu lắm"*, *"Chất lượng ổn nhưng wifi yếu"*, trong khi chỉ nhầm 5/40 câu (như thành ngữ *"Không có gì để chê"* hoặc câu trung tính về thời tiết *"Hôm nay trời mưa/đẹp"*).

---

## 5. Quy trình Cập nhật Hàng ngày & Demo (Tuần 3 & Tuần 4)

### 5.1 Pipeline Hàng ngày & Lập lịch Tự động (`scripts/03_daily_pipeline.py`, `03b_scheduler.py`)
- Hỗ trợ 3 chế độ thu thập (`replay` theo ngày thật, `json` nạp file mới kèm lọc PR + ẩn danh PII, và `playwright` công khai qua `03c_crawler_playwright.py`).
- Bảo đảm **tính bất biến (idempotent)** qua `daily_results/processed_ids.json`: chạy lại cùng ngày sẽ bỏ qua toàn bộ ID đã xử lý (`new_rows = 0`).
- **Kết quả kiểm thử 5 ngày liên tiếp thực tế (`2026-11-04` → `2026-11-08`):** Đã xử lý tổng cộng 66 bản ghi (`7`, `15`, `28`, `14`, `2` dòng/ngày), lưu thành `daily_results/sentiment_YYYY-MM-DD.csv`; chạy lại ngày `2026-11-06` xác nhận bỏ qua 28 bản ghi trùng (`new_rows = 0`); kiểm thử `schedule` (`03b_scheduler.py`) kích hoạt thành công và ghi log đầy đủ tại `daily_results/pipeline.log`.

### 5.2 Giao diện Demo Streamlit + Plotly (`app_demo.py`)
Ứng dụng web 4 tab cho phép chuyển đổi linh hoạt giữa 3 động cơ thực tế (`PhoBERT`, `TF-IDF+LR`, `TF-IDF+SVM`):
1. **Dự đoán trực tiếp:** Nhập câu hoặc chọn ví dụ nhanh; hiển thị văn bản sau ẩn danh/làm sạch, chủ đề, xác suất 3 lớp bằng Plotly, cảnh báo độ tin cậy thấp (`< 55%`), và Top-5 từ khóa đóng góp (khi dùng TF-IDF).
2. **Dashboard & Chủ đề:** Lọc theo loại (`post/comment`), nguồn, 6 chủ đề, nhãn, từ khóa; thẻ KPI; biểu đồ Donut & Grouped Bar theo chủ đề.
3. **Xu hướng Ngày/Tuần & Pipeline:** Chuyển đổi tổng hợp theo ngày/tuần, biểu đồ đường cảm xúc và chỉ số **Net Sentiment (%)**, kèm trình xem kết quả `daily_results/`.
4. **Đánh giá & Đạo đức:** Hiển thị bảng benchmark, biểu đồ Confusion Matrix/Benchmark, kết quả chống rò rỉ và cam kết bảo vệ PII.

---

## 6. Đạo đức AI, Pháp lý & Tính Tái lập (Mục 5.2d, 5.2e)

| Rủi ro đạo đức / pháp lý | Biện pháp giảm thiểu đã triển khai trong mã nguồn |
|---|---|
| **Lộ danh tính cá nhân (PII)** | Tách bỏ JSON thô khỏi repo/zip; băm SHA-256 `id`; xóa/che SĐT, email, IP, MSSV, URL, `@mention` trước khi lưu CSV; kiểm thử tự động `test_zero_residual_pii_in_all_data_csvs` xác nhận `0` PII sót. |
| **Thiên lệch dữ liệu (PR Bias & Imbalance)** | Loại bỏ 751 bài PR của Fanpage chính thức; áp dụng `class_weight="balanced"` và `CrossEntropyLoss` có trọng số lớp; đánh giá thêm trên Gold Real Test kiểm định tay. |
| **Lạm dụng diễn giải & Minh bạch** | Nêu rõ giới hạn dữ liệu (`semi_auto` vs `gold` kappa = `0.3500`), không dùng để đánh giá cá nhân, giữ giải thích từ khóa (`top_features`) cho mô hình tuyến tính. |
| **Tính chạy được & Tái lập (`seed=42`)** | Pin seed cố định ở tất cả bước; `requirements.txt` tương thích Python 3.10–3.13; 6 file `pytest` tự động kiểm chứng cả logic tiền xử lý, chống rò rỉ, mô hình, pipeline, tài liệu khớp JSON và khởi chạy Streamlit (`AppTest`). |

---

## 7. Hạn chế & Hướng Phát triển Tiếp theo

1. **Mở rộng quy mô & gán nhãn chéo nhóm:** Hiện tập sạch đạt 1868 mẫu (gần mục tiêu 3.000–6.000 mẫu nếu tiếp tục thu thập thêm từ các nhóm sinh viên công khai trong nhiều tháng); nhóm cần hoàn thiện gán nhãn chéo 200 mẫu (`data/annotation_batch_for_team.csv`) để fine-tune lại PhoBERT trực tiếp trên nhãn vàng do người gán (`kappa >= 0.70`).
2. **Xử lý thành ngữ phủ định cực đoan & câu ngoại miền:** Bổ sung dữ liệu câu hỏi trung tính / câu ngoại miền để khắc phục 5 trường hợp còn nhầm trên tập Challenge (như *"Không có gì để chê"*).
3. **Tối ưu triển khai sâu hơn:** Lượng tử hóa INT8 hoặc xuất ONNX cho PhoBERT nếu cần hạ độ trễ CPU từ `83.33 ms` xuống dưới `30 ms/câu` khi triển khai máy chủ không có GPU.
