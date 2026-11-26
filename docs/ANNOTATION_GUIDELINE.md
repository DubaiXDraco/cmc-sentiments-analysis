# Hướng dẫn Gán nhãn Cảm xúc Sinh viên CMC (Annotation Guideline)

Tài liệu này quy định chuẩn gán nhãn 3 lớp (`Positive`, `Neutral`, `Negative`) cho bài viết và bình luận công khai của sinh viên Trường Đại học CMC, phục vụ gán nhãn kiểm định nhóm (3–5 thành viên) và tính độ đồng thuận **Cohen's kappa**.

## 1. Quy ước 3 nhãn cảm xúc

| Nhãn | Tiếng Việt | Định nghĩa | Ví dụ điển hình |
|---|---|---|---|
| **Positive** | Tích cực | Thể hiện sự hài lòng, khen ngợi, hào hứng, tự hào, biết ơn, động viên tích cực đối với học tập, giảng viên, cơ sở vật chất, CLB hoặc đời sống sinh viên. | *"Phòng lab máy mượt quá, thầy dạy AI tâm huyết cực"*<br>*"CLB chào tân tối qua cháy quá trời!"* |
| **Neutral** | Bình thường | Hỏi đáp thông tin, thông báo trung tính, tìm đồ thất lạc, tìm bạn ở ghép/phòng trọ không phàn nàn, bình luận ngắn không mang sắc thái khen/chê rõ rệt. | *"Cho mình hỏi lịch thi cuối kỳ xem ở đâu vậy?"*<br>*"Mình để quên thẻ SV ở phòng 403, ai nhặt được cho xin lại"* |
| **Negative** | Tiêu cực | Than phiền, bực bội, thất vọng, lo lắng/sợ hãi tiêu cực, chê bai chất lượng dịch vụ, học phí, hệ thống CNTT, thái độ phục vụ hoặc cảnh báo bất tiện. | *"Web đăng ký môn học lag từ sáng không đăng ký được"*<br>*"Học phí tăng mà wifi tầng 4 chập chờn suốt"* |

## 2. Quy tắc xử lý các ca dễ nhầm lẫn

1. **Bài đăng tiện ích (Tìm đồ & Nhà trọ):**
   - Chỉ đăng tin tìm đồ thất lạc / trả đồ nhặt được / tìm bạn ở ghép với mục đích thông báo → **Neutral** (ngay cả khi có emoji khóc `😭` vì rơi đồ).
   - Nếu kèm phàn nàn chất lượng phòng trọ, mất cắp, thái độ dịch vụ → **Negative**.
2. **Câu phủ định & mỉa mai (Sarcasm):**
   - *"Không hài lòng với cách tổ chức"*, *"Không recommend"* → **Negative**.
   - *"Không có gì để chê"* → **Positive**.
   - Cần đọc toàn bộ câu, không phán đoán theo từ khóa đơn lẻ.
3. **Câu hỗn hợp (Mixed sentiment):**
   - Ưu tiên sắc thái chủ đạo hoặc vế kết luận (*"Phòng học đẹp nhưng đăng ký môn lag quá"* → **Negative**; *"Bài tập hơi nhiều nhưng thầy hỗ trợ rất nhiệt tình"* → **Positive**).
4. **Bài đăng tuyển thành viên CLB:**
   - Lời kêu gọi hào hứng, chào đón tân sinh viên → **Positive**.
   - Chỉ nêu yêu cầu kỹ thuật khô khan (*"Yêu cầu các bạn casting mỗi bạn nhảy 10p"*) → **Neutral**.

## 3. Quy trình kiểm định nhóm & tính Cohen's kappa

1. Mở file `data/annotation_batch_for_team.csv` (200 mẫu phân tầng từ tập dữ liệu thật đã ẩn danh).
2. Hai thành viên nhóm gán độc lập vào cột `annotator_1` và `annotator_2` mà **không nhìn cột `suggested_label`** trước khi chốt.
3. Chạy kiểm tra độ đồng thuận:
   ```bash
   python scripts/01c_cohen_kappa.py --file data/annotation_batch_for_team.csv --col-a annotator_1 --col-b annotator_2
   ```
4. Mục tiêu chấp nhận: **Cohen's kappa >= 0.70** (mức Substantial agreement); các mẫu bất đồng sẽ được nhóm thảo luận thống nhất nhãn cuối cùng.
