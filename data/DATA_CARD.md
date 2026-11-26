# Data Card – Bộ dữ liệu Cảm xúc Sinh viên Trường Đại học CMC

## 1. Nguồn gốc & Giấy phép sử dụng
- **Nguồn thu thập:** Dữ liệu bài đăng và bình luận công khai liên quan đến cộng đồng sinh viên Trường Đại học CMC từ 3 bản xuất Facebook công khai (ngày `2025-11-29` đến `2026-11-08`), gồm:
  - 2 hội nhóm sinh viên công khai: *Khu Tự Trị CMC* (1146 bản ghi sạch) và *Cộng Đồng Sinh Viên CMC* (422 bản ghi sạch).
  - Bình luận công khai của sinh viên trên 2 Fanpage chính thức: *ctsv.cmcuni* (213 bình luận) và *TruongDaiHocCMC* (87 bình luận).
- **Lọc nhiễu truyền thông:** Toàn bộ **751 bài đăng gốc (`post`) mang tính thông báo/PR** từ 2 Fanpage chính thức đã được loại bỏ khỏi tập dữ liệu để chỉ giữ lại tiếng nói thực của sinh viên.
- **Giấy phép & mục đích sử dụng:** Sử dụng nội bộ phục vụ **nghiên cứu học thuật và bài tập lớn môn AI/ML** tại Trường Đại học CMC (không thương mại hóa, không tái phân phối dữ liệu thô).

## 2. Quy trình Ẩn danh hóa & Bảo vệ Quyền riêng tư (PII)
- **Không lưu trữ JSON thô trong mã nguồn phát hành:** Các file JSON gốc bị loại trừ qua `.gitignore` và không nằm trong gói nộp bài.
- **Ẩn danh hóa trước khi lưu CSV (`src/anonymize.py`):**
  - Toàn bộ `id` gốc được băm SHA-256 có salt thành mã ẩn danh `cmc_<12hex>`.
  - Xóa URL, tên tác giả và link hồ sơ; thay thế SĐT Việt Nam (`[PHONE]`), email (`[EMAIL]`), IPv4 (`[IP]`), mã sinh viên (`[STUDENT_ID]`), `@mention` (`[USER]`) và tên sau cụm liên hệ (`[NAME]`).
  - Kiểm tra tự động bằng `contains_pii()` xác nhận **0 dòng sót PII** (`pii_residual_count = 0`) trên toàn bộ các file CSV trong `data/`.
- **Lưu ý hạn chế:** Các tên riêng được nhắc tự nhiên bằng văn bản thuần không có `@` trong bình luận giao lưu bạn bè vẫn tồn tại dưới dạng chuỗi văn bản công khai thông thường.

## 3. Thống kê Phễu Dữ liệu & Phân chia Tập (Không rò rỉ)
| Giai đoạn | Số bản ghi |
|---|---|
| Tổng mục thô quét từ 3 file JSON | 2763 |
| Loại bài đăng PR từ Fanpage chính thức | -751 |
| Loại văn bản trống | -10 |
| Bản ghi sinh viên không rỗng | 2002 |
| Sau lọc độ dài (`>= 5` ký tự sạch và `>= 2` từ) | 1936 |
| Sau loại trùng `text_clean` (`data/cmc_student_clean.csv`) | **1868** (1174 comment / 694 post) |
| Sau loại trùng câu gốc `core_text` (`GroupShuffleSplit`) | **1860** |
| **Train / Val / Test (70% / 15% / 15%)** | **1302 / 279 / 279** |

- **Kiểm tra rò rỉ (`core_text`):** `train_val_overlap = 0`, `train_test_overlap = 0`, `val_test_overlap = 0`.
- **Phân bố nhãn (1860 mẫu dedup):** `Neutral 915` (49.2%) / `Positive 636` (34.2%) / `Negative 309` (16.6%). Tập dữ liệu mất cân bằng theo lớp `Negative`, được xử lý khi huấn luyện bằng `class_weight="balanced"` (Baseline) và `CrossEntropyLoss(weight=[1.9908, 0.6878, 0.9581])` (PhoBERT).
- **Phân bố 6 chủ đề (1868 mẫu):** Giao lưu & Khác (1301), Hoạt động & CLB (207), Học vụ & Giảng viên (139), Đời sống & Nhà trọ (103), Cơ sở vật chất & CNTT (71), Tìm đồ & Hỗ trợ (47).

## 4. Quy trình Gán nhãn & Kiểm định Chất lượng
1. **Gán nhãn bán tự động (`scripts/01b_semi_auto_label.py`):**
   - Dùng `wonrax/phobert-base-vietnamese-sentiment` suy luận trên 1868 mẫu (`1837` mẫu từ PhoBERT + `31` mẫu bài tiện ích tìm đồ/nhà trọ không phàn nàn chuyển sang `Neutral`).
2. **Kiểm định thủ công tập thật (`data/gold_real_test.csv`):**
   - Rút mẫu phân tầng **120 bản ghi thật** hoàn toàn từ `data/test.csv` (40 mỗi lớp bán tự động, `seed=42`, 0 giao với Train/Val) và rà soát gán nhãn thủ công từng câu theo `docs/ANNOTATION_GUIDELINE.md`.
   - Phân bố nhãn chuẩn (`gold_label`): `Neutral 80 / Positive 30 / Negative 10`.
   - Độ đồng thuận giữa nhãn bán tự động và nhãn kiểm định tay: **56.67% (`0.5667`)**, **Cohen's kappa = `0.3500`** (đã chỉnh sửa 52/120 mẫu). Điều này chứng minh sự lệch miền (domain drift) của mô hình gán nhãn tổng quát trước tiếng lóng, tin ghép trọ và câu hỏi của sinh viên CMC.
3. **Bộ gán nhãn nhóm & Bộ thử thách độc lập:**
   - `data/annotation_batch_for_team.csv`: 200 mẫu chuẩn bị cho 2 thành viên gán nhãn độc lập và đo Cohen's kappa bằng `scripts/01c_cohen_kappa.py`.
   - `data/challenge_test_set.csv`: 40 câu viết tay kiểm tra phủ định, hỗn hợp và ngoài chủ đề (0/40 câu trùng Train/Val).
