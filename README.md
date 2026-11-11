# 📊 CMC University Student Sentiment Analysis (`cmc-sentiments-analysis`)

Mô hình phân tích cảm xúc sinh viên Trường Đại học CMC dựa trên bình luận và bài đăng công khai trên Facebook (Fanpage trường, CMCU Confessions, Hội nhóm sinh viên).

## 📁 Cấu trúc thư mục dự án
```text
cmc-sentiments-analysis/
├── data/
│   ├── raw/                  # Dữ liệu thô cào từ Facebook (cmc_fb_raw.csv)
│   └── processed/            # Dữ liệu đã làm sạch, tách từ, chia tập train/val/test & ảnh EDA
├── models/                   # Lưu mô hình Baseline (.pkl) & PhoBERT
├── logs/                     # Lưu log chạy pipeline tự động hàng ngày (Tuần 3)
├── src/
│   ├── crawl_fb_console.js   # Script JS cào bình luận Facebook trực tiếp trên trình duyệt
│   ├── data_pipeline.py      # Làm sạch, xử lý teencode, tách từ PhoBERT & chia Train/Val/Test
│   ├── eda_generator.py      # Tự động xuất biểu đồ phân tích khám phá dữ liệu (EDA)
│   ├── train_baseline.py     # Huấn luyện & đánh giá TF-IDF + Logistic Regression
│   └── daily_updater.py      # Pipeline tự động cập nhật & dự đoán hàng ngày (schedule)
├── app.py                    # Giao diện Demo Streamlit (Tuần 4)
└── requirements.txt          # Danh sách thư viện cài đặt
```

## 🚀 Hướng dẫn chạy nhanh (Quickstart)

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy Pipeline Tiền xử lý & Chia tập Train/Val/Test (Tuần 1 & 2)
```bash
python src/data_pipeline.py
python src/eda_generator.py
```

### 3. Huấn luyện mô hình Baseline (Tuần 2)
```bash
python src/train_baseline.py
```

### 4. Chạy Pipeline cập nhật tự động hàng ngày (Tuần 3)
```bash
python src/daily_updater.py
```

### 5. Khởi chạy Giao diện Demo (Tuần 4)
```bash
streamlit run app.py
```

## 🛡️ Đạo đức dữ liệu & Quyền riêng tư (Data Privacy)
* Chỉ thu thập dữ liệu văn bản từ các bài đăng/bình luận công khai.
* Toàn bộ dữ liệu được ẩn danh hoàn toàn (loại bỏ tên tài khoản, UID, email, số điện thoại) trước khi đưa vào lưu trữ và huấn luyện.
