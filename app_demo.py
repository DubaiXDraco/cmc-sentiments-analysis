#!/usr/bin/env python3
"""
app_demo.py
Tuần 4: Giao diện Demo Streamlit + Plotly cho Hệ thống Phân tích Cảm xúc Sinh viên CMC.
Hỗ trợ 3 động cơ ML/NLP thực tế (Fine-tuned PhoBERT, TF-IDF + LR, TF-IDF + LinearSVC),
4 tab: Dự đoán trực tiếp, Dashboard & Chủ đề, Xu hướng Ngày/Tuần & Pipeline, Đánh giá & Đạo đức.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import LABELS, LABEL2VI, SentimentEngine

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DAILY_DIR = ROOT / "daily_results"
MODELS_DIR = ROOT / "models"

COLOR_MAP: Dict[str, str] = {
    "Positive": "#10B981",
    "Neutral": "#3B82F6",
    "Negative": "#EF4444",
}
EMOJI_MAP: Dict[str, str] = {
    "Positive": "😊",
    "Neutral": "😐",
    "Negative": "😟",
}
QUICK_EXAMPLES = [
    "CSVC trường mik xịn sò thực sự, phòng lab máy cấu hình cao học AI mượt lun!",
    "Web dkmh sáng nay lag quá, quay đều từ sáng ko đk đc môn nào hết huhu",
    "Cho mik hỏi lịch thi cuối kỳ học phần Toán rời rạc xem ở đâu z mn?",
    "Không hài lòng với cách tổ chức sự kiện hôm nay lắm",
    "Phòng học đẹp nhưng wifi tầng 4 chập chờn quá",
]

st.set_page_config(
    page_title="CMC Student Sentiment Analysis",
    page_icon="🎓",
    layout="wide",
)


def _phobert_available() -> bool:
    pdir = MODELS_DIR / "cmc_phobert_sentiment"
    return (pdir / "config.json").exists() and (
        (pdir / "model.safetensors").exists() or (pdir / "pytorch_model.bin").exists()
    )


@st.cache_resource(show_spinner="Đang nạp mô hình phân tích cảm xúc...")
def get_engine(backend: str) -> SentimentEngine:
    return SentimentEngine(backend=backend, models_dir=MODELS_DIR)


@st.cache_data(show_spinner="Đang nạp dữ liệu cộng đồng sinh viên CMC đã ẩn danh...")
def load_labeled_data() -> pd.DataFrame:
    path = DATA_DIR / "cmc_student_labeled.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    if "likes" in df.columns:
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_eval_json() -> dict:
    path = REPORTS_DIR / "evaluation_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_ingestion_stats() -> dict:
    path = REPORTS_DIR / "ingestion_stats.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def set_example_text(text: str) -> None:
    st.session_state["user_text"] = text


def render_sidebar(comp_df: pd.DataFrame, df_all: pd.DataFrame) -> SentimentEngine:
    st.sidebar.header("⚙️ Cấu hình Mô hình")
    options = []
    if _phobert_available():
        options.append(("phobert", "Fine-tuned PhoBERT (vinai/phobert-base-v2)"))
    if (MODELS_DIR / "model_tfidf_lr.joblib").exists():
        options.append(("tfidf_lr", "TF-IDF + Logistic Regression"))
    if (MODELS_DIR / "model_tfidf_svm.joblib").exists():
        options.append(("tfidf_svm", "TF-IDF (Word+Char) + LinearSVC"))

    if not options:
        st.sidebar.error("Chưa tìm thấy mô hình huấn luyện trong thư mục `models/`.")
        st.stop()

    key_to_label = dict(options)
    selected_key = st.sidebar.selectbox(
        "Chọn mô hình suy luận:",
        options=[k for k, _ in options],
        format_func=lambda k: key_to_label[k],
    )
    engine = get_engine(selected_key)
    st.sidebar.success(f"**Engine:** {engine.backend_status}")

    if not comp_df.empty and selected_key in set(comp_df["model_key"]):
        row = comp_df[comp_df["model_key"] == selected_key].iloc[0]
        st.sidebar.markdown(
            f"""
            - **Gold Test Accuracy:** `{row['gold_acc']:.4f}`
            - **Gold Test Macro-F1:** `{row['gold_macro_f1']:.4f}`
            - **Độ trễ CPU TB:** `{row['latency_ms']:.2f} ms/câu`
            - **Dung lượng:** `{row['size_mb']:.2f} MB`
            """
        )

    st.sidebar.divider()
    st.sidebar.markdown("### 📦 Bộ dữ liệu thật (đã ẩn danh)")
    st.sidebar.info(
        f"- Tổng bản ghi sạch: **{len(df_all):,}** (Group SV + bình luận SV)\n"
        "- Đã loại bài PR từ Fanpage trường\n"
        "- Đã ẩn danh ID, SĐT, email, MSSV, URL"
    )
    st.sidebar.caption(f"Cập nhật giao diện: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return engine


def render_tab_predict(engine: SentimentEngine) -> None:
    st.subheader("🤖 Dự đoán Cảm xúc & Chủ đề Trực tiếp")
    st.caption(
        "Văn bản nhập vào sẽ đi qua pipeline thống nhất: Ẩn danh hóa PII → Chuẩn hóa Unicode & Teencode → Phân loại chủ đề → Suy luận ML/PhoBERT."
    )

    user_input = st.text_area(
        "Nhập bình luận hoặc bài viết sinh viên CMC:",
        key="user_text",
        height=120,
        placeholder="Ví dụ: Wifi tầng 4 hơi chậm nhưng thầy dạy môn AI rất tâm huyết...",
    )
    predict_clicked = st.button("Phân tích ngay", type="primary")

    if predict_clicked:
        if not user_input.strip():
            st.warning("Vui lòng nhập nội dung văn bản trước khi phân tích.")
        else:
            res = engine.predict(user_input)
            c1, c2 = st.columns([1.15, 1.0])
            with c1:
                color = COLOR_MAP.get(res.label, "#3B82F6")
                emoji = EMOJI_MAP.get(res.label, "💬")
                st.markdown(f"### Kết quả: {emoji} **{res.label} ({res.label_vi})**")
                st.progress(min(max(res.confidence, 0.0), 1.0))
                st.markdown(
                    f"**Chủ đề phát hiện:** `{res.topic}`  |  "
                    f"**Độ tự tin:** `{res.confidence * 100:.2f}%`  |  "
                    f"**Độ trễ:** `{res.latency_ms:.2f} ms`"
                )
                if res.confidence < 0.55:
                    st.info("Độ tự tin thấp (< 55%): câu có sắc thái trung tính/hỗn hợp hoặc mô hình chưa chắc chắn.")

                with st.expander("Xem văn bản sau ẩn danh & tiền xử lý"):
                    st.write("**Sau ẩn danh PII:**", res.anonymized_text or "(trống)")
                    st.code(res.cleaned_text or "(trống)", language="text")

                if res.top_features:
                    feat_str = ", ".join(f"`{f}` ({w:+.3f})" for f, w in res.top_features)
                    st.markdown(f"**Từ khóa đóng góp chính (Explainability):** {feat_str}")

            with c2:
                prob_df = pd.DataFrame(
                    {
                        "Cảm xúc": LABELS,
                        "Xác suất (%)": [round(res.probabilities[lbl] * 100, 2) for lbl in LABELS],
                    }
                )
                fig = px.bar(
                    prob_df,
                    x="Xác suất (%)",
                    y="Cảm xúc",
                    orientation="h",
                    color="Cảm xúc",
                    color_discrete_map=COLOR_MAP,
                    range_x=[0, 100],
                    text_auto=".2f",
                    title="Phân bố xác suất 3 lớp cảm xúc",
                )
                fig.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Ví dụ nhanh (bấm để điền vào ô nhập):**")
    for idx, ex in enumerate(QUICK_EXAMPLES):
        st.button(ex, key=f"quick_ex_{idx}", on_click=set_example_text, args=(ex,))


def render_tab_dashboard(df: pd.DataFrame) -> None:
    st.subheader("📊 Dashboard Phân tích Cộng đồng Sinh viên CMC")
    if df.empty:
        st.warning("Chưa có dữ liệu `data/cmc_student_labeled.csv`.")
        return

    f1, f2, f3, f4, f5 = st.columns([1.0, 1.2, 1.3, 1.0, 1.4])
    with f1:
        sel_type = st.selectbox("Loại nội dung:", ["Tất cả", "post", "comment"])
    with f2:
        sources = ["Tất cả"] + sorted(df["source"].dropna().unique().tolist())
        sel_source = st.selectbox("Nguồn dữ liệu:", sources)
    with f3:
        topics = ["Tất cả"] + sorted(df["topic"].dropna().unique().tolist())
        sel_topic = st.selectbox("Chủ đề:", topics)
    with f4:
        sel_label = st.selectbox("Nhãn cảm xúc:", ["Tất cả"] + LABELS)
    with f5:
        kw = st.text_input("Tìm từ khóa:", placeholder="vd: wifi, học phí, clb...")

    fdf = df.copy()
    if sel_type != "Tất cả":
        fdf = fdf[fdf["type"] == sel_type]
    if sel_source != "Tất cả":
        fdf = fdf[fdf["source"] == sel_source]
    if sel_topic != "Tất cả":
        fdf = fdf[fdf["topic"] == sel_topic]
    if sel_label != "Tất cả":
        fdf = fdf[fdf["label"] == sel_label]
    if kw.strip():
        q = kw.strip().lower()
        fdf = fdf[fdf["text"].fillna("").str.lower().str.contains(q, na=False)]

    total = len(fdf)
    pos = int((fdf["label"] == "Positive").sum())
    neu = int((fdf["label"] == "Neutral").sum())
    neg = int((fdf["label"] == "Negative").sum())
    likes_sum = int(fdf["likes"].sum()) if "likes" in fdf.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tổng bản ghi", f"{total:,}", f"{int((fdf['type']=='post').sum())} post | {int((fdf['type']=='comment').sum())} cmt")
    k2.metric("😊 Positive", f"{pos:,}", f"{(pos / total * 100) if total else 0:.1f}%")
    k3.metric("😐 Neutral", f"{neu:,}", f"{(neu / total * 100) if total else 0:.1f}%")
    k4.metric("😟 Negative", f"{neg:,}", f"{(neg / total * 100) if total else 0:.1f}%")
    k5.metric("👍 Tổng lượt thích", f"{likes_sum:,}", f"TB {(likes_sum / total) if total else 0:.1f}/mẫu")

    st.divider()
    c1, c2 = st.columns([1.0, 1.4])
    with c1:
        if total > 0:
            pie_df = fdf["label"].value_counts().reindex(LABELS, fill_value=0).reset_index()
            pie_df.columns = ["Nhãn", "Số lượng"]
            pie_df = pie_df[pie_df["Số lượng"] > 0]
            fig_pie = px.pie(
                pie_df,
                names="Nhãn",
                values="Số lượng",
                color="Nhãn",
                color_discrete_map=COLOR_MAP,
                hole=0.45,
                title="Tỷ lệ 3 nhãn cảm xúc",
            )
            fig_pie.update_traces(textinfo="percent+label+value")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Không có bản ghi phù hợp bộ lọc.")

    with c2:
        if total > 0:
            topic_df = fdf.groupby(["topic", "label"]).size().reset_index(name="Số lượng")
            fig_bar = px.bar(
                topic_df,
                x="topic",
                y="Số lượng",
                color="label",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                category_orders={"label": LABELS},
                text_auto=True,
                title="Phân bố cảm xúc theo 6 chủ đề sinh viên",
            )
            fig_bar.update_layout(xaxis_title="Chủ đề", yaxis_title="Số lượng")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader(f"📋 Danh sách bài viết & bình luận đã ẩn danh ({total:,} dòng)")
    show_cols = [c for c in ["id", "type", "source", "date", "topic", "text", "likes", "label", "label_source"] if c in fdf.columns]
    st.dataframe(fdf[show_cols].head(500), use_container_width=True, hide_index=True, height=380)


def render_tab_trends(df: pd.DataFrame) -> None:
    st.subheader("📈 Xu hướng Cảm xúc Theo Ngày / Tuần & Pipeline Hàng ngày")
    if df.empty or "date_dt" not in df.columns:
        st.warning("Chưa có dữ liệu thời gian để hiển thị xu hướng.")
        return

    granularity = st.radio("Khung thời gian tổng hợp:", ["Theo Tuần (Weekly)", "Theo Ngày (Daily)"], horizontal=True)
    freq = "W" if "Tuần" in granularity else "D"

    tdf = df.dropna(subset=["date_dt"]).copy()
    tdf["period"] = tdf["date_dt"].dt.to_period(freq).apply(lambda p: p.start_time)
    grouped = tdf.groupby(["period", "label"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    grouped["Total"] = grouped.sum(axis=1)
    grouped["Net_Sentiment_Pct"] = ((grouped["Positive"] - grouped["Negative"]) / grouped["Total"].replace(0, 1)) * 100.0
    grouped = grouped.reset_index()

    c1, c2 = st.columns(2)
    with c1:
        melted = grouped.melt(id_vars="period", value_vars=LABELS, var_name="Cảm xúc", value_name="Số lượng")
        fig_line = px.line(
            melted,
            x="period",
            y="Số lượng",
            color="Cảm xúc",
            color_discrete_map=COLOR_MAP,
            markers=True,
            title=f"Số lượng bài viết/bình luận theo thời gian ({freq})",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        fig_net = go.Figure()
        fig_net.add_trace(
            go.Bar(
                x=grouped["period"],
                y=grouped["Net_Sentiment_Pct"],
                marker_color=["#10B981" if v >= 0 else "#EF4444" for v in grouped["Net_Sentiment_Pct"]],
                name="Net Sentiment (%)",
            )
        )
        fig_net.update_layout(
            title="Chỉ số Cảm xúc Ròng: (Positive - Negative) / Total (%)",
            xaxis_title="Thời gian",
            yaxis_title="Net Sentiment (%)",
        )
        st.plotly_chart(fig_net, use_container_width=True)

    st.divider()
    st.markdown("### 🔄 Kết quả chạy Pipeline Hàng ngày (`daily_results/`)")
    daily_files = sorted(DAILY_DIR.glob("sentiment_*.csv")) if DAILY_DIR.exists() else []
    if daily_files:
        sel_file = st.selectbox("Chọn file kết quả theo ngày:", daily_files, index=len(daily_files) - 1, format_func=lambda p: p.name)
        ddf = pd.read_csv(sel_file, encoding="utf-8-sig")
        st.write(f"File `{sel_file.name}` gồm **{len(ddf)}** bản ghi:")
        st.dataframe(ddf.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có file kết quả nào trong `daily_results/`.")

    log_path = DAILY_DIR / "pipeline.log"
    if log_path.exists():
        with st.expander("Xem nhật ký chạy Pipeline (`daily_results/pipeline.log`)"):
            st.code(log_path.read_text(encoding="utf-8")[-3000:], language="text")


def render_tab_evaluation(comp_df: pd.DataFrame, eval_json: dict, ingestion_stats: dict) -> None:
    st.subheader("🧪 So sánh Mô hình, Kiểm soát Rò rỉ & Đạo đức Dữ liệu")
    if not comp_df.empty:
        st.markdown("#### 1. Bảng đối chiếu hiệu năng & tốc độ CPU thực tế")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    bm_img = REPORTS_DIR / "model_benchmark.png"
    cm_img = REPORTS_DIR / "confusion_matrix_comparison.png"
    if bm_img.exists():
        c1.image(str(bm_img), caption="Benchmark Accuracy / Macro-F1 & Độ trễ CPU", use_container_width=True)
    if cm_img.exists():
        c2.image(str(cm_img), caption="Confusion Matrix trên Gold Real Test (120 mẫu thật kiểm định tay)", use_container_width=True)

    if eval_json:
        st.markdown(
            f"- **Chống rò rỉ (`core_text` + `GroupShuffleSplit`):** "
            f"`Train={eval_json.get('n_train')}`, `Val={eval_json.get('n_val')}`, `Test={eval_json.get('n_test')}`, "
            f"trùng lặp giữa các tập = `{eval_json.get('leakage_check')}`"
        )
        st.markdown(
            f"- **Gold Real Test (120 mẫu thật từ tập Test được kiểm định thủ công):** `{eval_json.get('n_gold_real_test')}` mẫu | "
            f"**Challenge Test (câu phủ định/hỗn hợp/ngoài miền độc lập):** `{eval_json.get('n_challenge_test')}` mẫu"
        )

    excluded_pr = ingestion_stats.get("excluded_official_fanpage_posts", 0)
    st.divider()
    st.markdown(
        f"""
        #### 2. Đạo đức AI, Pháp lý & Tính minh bạch
        - **Không lưu trữ PII:** Toàn bộ ID bài đăng/bình luận gốc đã băm SHA-256 có salt; SĐT, email, URL, IP, MSSV và @mention đã được ẩn danh trước khi lưu CSV. File JSON thô không nằm trong mã nguồn phát hành.
        - **Lọc nhiễu PR chính thức:** {excluded_pr:,} bài đăng thông báo/truyền thông từ Fanpage chính thức (`TruongDaiHocCMC`, `ctsv.cmcuni`) được tách khỏi tập huấn luyện để không làm sai lệch cảm xúc thực của sinh viên.
        - **Gán nhãn minh bạch:** Sử dụng gán nhãn bán tự động bằng `wonrax/phobert-base-vietnamese-sentiment` kết hợp quy tắc tiện ích (tìm đồ/nhà trọ) và kiểm định thủ công 120 mẫu thật (`data/gold_real_test.csv`).
        """
    )


def main() -> None:
    st.title("🎓 Hệ thống Phân tích Cảm xúc Sinh viên Trường Đại học CMC")
    st.caption("Kết hợp Dữ liệu cộng đồng CMC thật đã ẩn danh • Baseline TF-IDF & Fine-tuned PhoBERT • Pipeline hàng ngày & Kiểm soát rò rỉ")

    df_all = load_labeled_data()
    comp_df = load_model_comparison()
    eval_json = load_eval_json()
    ingestion_stats = load_ingestion_stats()
    engine = render_sidebar(comp_df, df_all)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🤖 Dự đoán trực tiếp",
            "📊 Dashboard & Chủ đề",
            "📈 Xu hướng Ngày/Tuần & Pipeline",
            "🧪 Đánh giá & Đạo đức",
        ]
    )
    with tab1:
        render_tab_predict(engine)
    with tab2:
        render_tab_dashboard(df_all)
    with tab3:
        render_tab_trends(df_all)
    with tab4:
        render_tab_evaluation(comp_df, eval_json, ingestion_stats)


if __name__ == "__main__":
    main()
