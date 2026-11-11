import streamlit as st
import pandas as pd
import joblib
from src.data_pipeline import clean_vietnamese_text

st.set_page_config(page_title="CMC Sentiment Analysis", layout="wide")
st.title("📊 Hệ thống Phân tích Cảm xúc Sinh viên Đại học CMC")

model = joblib.load("models/sentiment_baseline.pkl")
df = pd.read_csv("data/processed/cmc_clean_labeled.csv")
label_names = {0: "Negative (Tiêu cực) 😡", 1: "Neutral (Trung lập) 😐", 2: "Positive (Tích cực) 😊"}

st.subheader("🔍 1. Dự đoán Cảm xúc Trực tiếp từ Văn bản")
user_input = st.text_area("Nhập bình luận hoặc bài đăng:", "Thư viện trường mình mát, nhiều ổ điện ngồi học rất thích!")
if st.button("Phân tích cảm xúc"):
    cleaned = clean_vietnamese_text(user_input)
    pred = model.predict([cleaned])[0]
    probs = model.predict_proba([cleaned])[0]
    st.success(f"**Kết quả:** {label_names[pred]} (Độ tin cậy: {max(probs)*100:.1f}%)")
    st.caption(f"Văn bản sau tiền xử lý: `{cleaned}`")

st.subheader("📈 2. Thống kê & Xu hướng Cảm xúc theo Thời gian")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Phân bố cảm xúc tổng thể**")
    st.bar_chart(df["sentiment"].value_counts())
with col2:
    st.markdown("**Xu hướng cảm xúc theo ngày**")
    trend_df = df.groupby(["timestamp", "sentiment"]).size().unstack(fill_value=0)
    st.line_chart(trend_df)
