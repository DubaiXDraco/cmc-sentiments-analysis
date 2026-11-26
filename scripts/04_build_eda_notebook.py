#!/usr/bin/env python3
"""
scripts/04_build_eda_notebook.py
Tuần 1 & 4: Tạo notebook `notebooks/01_preprocess_eda.ipynb` và thực thi trực tiếp
tuần tự các cell trong tiến trình Python (không dùng ZMQ kernel) để ghi output
vào notebook và xuất 5 biểu đồ EDA vào `reports/`.
"""

import ast
import contextlib
import io
import os
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "01_preprocess_eda.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            "# 📊 Notebook 01: Tiền xử lý, Ẩn danh hóa & Khám phá Dữ liệu (EDA)\n"
            "**Dự án:** Phân tích Cảm xúc Sinh viên Trường Đại học CMC  \n"
            "**Nguồn dữ liệu:** Bình luận & bài đăng công khai từ cộng đồng sinh viên CMC "
            "(đã loại bài PR chính thức từ Fanpage trường và ẩn danh hóa PII trước khi lưu CSV)."
        ),
        new_code_cell(
            "import json\n"
            "from collections import Counter\n"
            "from pathlib import Path\n"
            "import sys\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.anonymize import contains_pii\n"
            "from src.preprocess import clean_text\n\n"
            "DATA_DIR = ROOT / 'data'\n"
            "REPORTS_DIR = ROOT / 'reports'\n"
            "REPORTS_DIR.mkdir(exist_ok=True)\n"
            "plt.rcParams['figure.dpi'] = 130"
        ),
        new_markdown_cell("## 1. Kiểm tra phễu thu thập, lọc nhiễu & ẩn danh hóa PII"),
        new_code_cell(
            "stats = json.loads((REPORTS_DIR / 'ingestion_stats.json').read_text(encoding='utf-8'))\n"
            "split_stats = json.loads((REPORTS_DIR / 'split_stats.json').read_text(encoding='utf-8'))\n"
            "print('--- THỐNG KÊ PHỄU DỮ LIỆU THẬT ---')\n"
            "for k in ['n_json_files', 'total_raw_items_scanned', 'excluded_official_fanpage_posts', "
            "'student_nonempty_records', 'after_length_filter', 'after_exact_text_clean_dedup', "
            "'unique_core_text_count', 'pii_residual_count']:\n"
            "    print(f'{k:35s}: {stats[k]}')\n"
            "print('Train / Val / Test (dedup core):', split_stats['n_train'], '/', split_stats['n_val'], '/', split_stats['n_test'])"
        ),
        new_markdown_cell("## 2. Đọc dữ liệu đã làm sạch & gán nhãn bán tự động"),
        new_code_cell(
            "df = pd.read_csv(DATA_DIR / 'cmc_student_labeled.csv', encoding='utf-8-sig')\n"
            "df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')\n"
            "df['char_len'] = df['text_clean'].fillna('').str.len()\n"
            "df['word_len'] = df['text_clean'].fillna('').str.split().str.len()\n"
            "assert int(df['text'].fillna('').apply(contains_pii).sum()) == 0\n"
            "print('Tổng mẫu sạch:', len(df))\n"
            "print('Phân bố nhãn:', df['label'].value_counts().to_dict())\n"
            "print('Độ dài TB (từ):', round(df['word_len'].mean(), 2), '| Trung vị:', df['word_len'].median())\n"
            "df[['id', 'type', 'source', 'date', 'topic', 'label', 'text_clean']].head(5)"
        ),
        new_markdown_cell("## 3. Biểu đồ phân bố cảm xúc & nguồn/loại nội dung"),
        new_code_cell(
            "COLORS = {'Positive': '#10B981', 'Neutral': '#3B82F6', 'Negative': '#EF4444'}\n"
            "order = ['Positive', 'Neutral', 'Negative']\n"
            "counts = df['label'].value_counts().reindex(order, fill_value=0)\n\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "bars = ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index], edgecolor='black')\n"
            "ax.set_title('Phân bố Cảm xúc Cộng đồng Sinh viên CMC (Dữ liệu thật đã ẩn danh)', fontweight='bold')\n"
            "ax.set_ylabel('Số lượng bản ghi')\n"
            "for b in bars:\n"
            "    h = b.get_height()\n"
            "    ax.text(b.get_x() + b.get_width()/2, h, f'{int(h)} ({h/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')\n"
            "ax.grid(axis='y', linestyle='--', alpha=0.4)\n"
            "plt.tight_layout()\n"
            "plt.savefig(REPORTS_DIR / 'eda_sentiment_dist.png', dpi=150)\n"
            "plt.close(fig)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))\n"
            "src_cnt = df['source'].value_counts()\n"
            "axes[0].barh(src_cnt.index, src_cnt.values, color='#0EA5E9', edgecolor='black')\n"
            "axes[0].set_title('Phân bố theo Nguồn (bài Group + bình luận SV)', fontweight='bold')\n"
            "axes[0].set_xlabel('Số lượng')\n"
            "type_cnt = df['type'].value_counts()\n"
            "axes[1].bar(type_cnt.index, type_cnt.values, color=['#6366F1', '#14B8A6'], edgecolor='black')\n"
            "axes[1].set_title('Tỷ trọng Bài đăng (post) vs Bình luận (comment)', fontweight='bold')\n"
            "axes[1].set_ylabel('Số lượng')\n"
            "plt.tight_layout()\n"
            "plt.savefig(REPORTS_DIR / 'eda_source_type.png', dpi=150)\n"
            "plt.close(fig)\n"
            "print('Đã lưu eda_sentiment_dist.png và eda_source_type.png')"
        ),
        new_markdown_cell("## 4. Biểu đồ phân bố cảm xúc theo 6 chủ đề sinh viên"),
        new_code_cell(
            "ct = pd.crosstab(df['topic'], df['label']).reindex(columns=order, fill_value=0)\n"
            "ax = ct.plot(kind='barh', stacked=True, figsize=(10, 5), color=[COLORS[c] for c in order], edgecolor='black')\n"
            "ax.set_title('Phân bố Cảm xúc theo 6 Chủ đề Thảo luận của Sinh viên CMC', fontweight='bold')\n"
            "ax.set_xlabel('Số lượng bài đăng & bình luận')\n"
            "ax.set_ylabel('Chủ đề')\n"
            "ax.grid(axis='x', linestyle='--', alpha=0.4)\n"
            "plt.tight_layout()\n"
            "plt.savefig(REPORTS_DIR / 'eda_topic_sentiment.png', dpi=150)\n"
            "plt.close()\n"
            "ct"
        ),
        new_markdown_cell("## 5. Phân phối độ dài câu & xu hướng cảm xúc theo tuần"),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))\n"
            "clipped = df['word_len'].clip(upper=100)\n"
            "axes[0].hist(clipped, bins=25, color='#6366F1', edgecolor='white')\n"
            "axes[0].axvline(df['word_len'].mean(), color='#EF4444', linestyle='--', label=f\"TB: {df['word_len'].mean():.1f} từ\")\n"
            "axes[0].set_title('Phân phối độ dài câu (giới hạn 100 từ)', fontweight='bold')\n"
            "axes[0].set_xlabel('Số từ')\n"
            "axes[0].set_ylabel('Tần suất')\n"
            "axes[0].legend()\n\n"
            "sns.boxplot(data=df[df['word_len'] <= 80], x='label', y='word_len', order=order, hue='label', palette=COLORS, legend=False, ax=axes[1])\n"
            "axes[1].set_title('Độ dài câu theo Nhãn cảm xúc', fontweight='bold')\n"
            "axes[1].set_xlabel('Nhãn')\n"
            "axes[1].set_ylabel('Số từ')\n"
            "plt.tight_layout()\n"
            "plt.savefig(REPORTS_DIR / 'eda_length_dist.png', dpi=150)\n"
            "plt.close(fig)\n\n"
            "tdf = df.dropna(subset=['date_dt']).copy()\n"
            "tdf['week'] = tdf['date_dt'].dt.to_period('W').apply(lambda p: p.start_time)\n"
            "weekly = tdf.groupby(['week', 'label']).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)\n"
            "recent = weekly.tail(14)\n"
            "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "for lbl in order:\n"
            "    ax.plot(recent.index, recent[lbl], marker='o', label=lbl, color=COLORS[lbl], linewidth=2)\n"
            "ax.set_title('Xu hướng Cảm xúc Sinh viên CMC theo Tuần (14 tuần gần nhất)', fontweight='bold')\n"
            "ax.set_ylabel('Số lượng bản ghi')\n"
            "ax.legend()\n"
            "ax.grid(True, linestyle='--', alpha=0.4)\n"
            "plt.xticks(rotation=30)\n"
            "plt.tight_layout()\n"
            "plt.savefig(REPORTS_DIR / 'eda_weekly_trend.png', dpi=150)\n"
            "plt.close(fig)\n"
            "print('Đã lưu eda_length_dist.png và eda_weekly_trend.png')"
        ),
        new_markdown_cell(
            "## 6. Kiểm tra chất lượng tiền xử lý & từ khóa phổ biến\n"
            "Xác nhận các lỗi teencode trước đây (`cmc -> đại học đại học`, `bt -> bình thường`, `k5`) đã được xử lý sạch."
        ),
        new_code_cell(
            "samples = [\n"
            "    'Trường Đại học CMC tuyển sinh K5 và K4',\n"
            "    'Ai bt lịch thi ck chưa, mình ko bt luôn',\n"
            "    'CSVC phòng lab xịn quáaaaa, liên hệ 0912345678 hoặc sv@cmc.edu.vn'\n"
            "]\n"
            "for s in samples:\n"
            "    print('Gốc  :', s)\n"
            "    print('Sạch :', clean_text(s))\n"
            "    print('-'*40)\n\n"
            "top_words = Counter(' '.join(df['text_clean'].fillna('')).split()).most_common(15)\n"
            "print('Top 15 từ phổ biến:', top_words)"
        ),
    ]
    nb = new_notebook(cells=cells)
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def execute_notebook_in_process(nb: nbformat.NotebookNode, workdir: Path) -> None:
    """Thực thi các code cell tuần tự trong tiến trình hiện tại, không dùng ZMQ kernel."""
    prev_cwd = Path.cwd()
    namespace = {"__name__": "__main__"}
    exec_count = 0
    try:
        os.chdir(workdir)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            exec_count += 1
            cell.execution_count = exec_count
            outputs = []
            buf = io.StringIO()
            tree = ast.parse(cell.source, mode="exec")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(body=tree.body.pop().value)

            with contextlib.redirect_stdout(buf):
                if tree.body:
                    exec(compile(tree, filename=f"<cell-{exec_count}>", mode="exec"), namespace)
                last_val = None
                if last_expr is not None:
                    last_val = eval(compile(last_expr, filename=f"<cell-{exec_count}-expr>", mode="eval"), namespace)

            stdout_text = buf.getvalue()
            if stdout_text:
                outputs.append(new_output("stream", name="stdout", text=stdout_text))
            if last_val is not None:
                data = {"text/plain": repr(last_val)}
                if hasattr(last_val, "to_html"):
                    try:
                        data["text/html"] = last_val.to_html()
                    except Exception:
                        pass
                outputs.append(
                    new_output("execute_result", execution_count=exec_count, data=data, metadata={})
                )
            cell.outputs = outputs
    finally:
        os.chdir(prev_cwd)


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    execute_notebook_in_process(nb, NB_PATH.parent)
    with open(NB_PATH, mode="w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"✓ Đã tạo và thực thi in-process notebook hoàn chỉnh: {NB_PATH}")


if __name__ == "__main__":
    main()
