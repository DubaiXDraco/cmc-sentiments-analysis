import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_eda(clean_csv_path="data/processed/cmc_clean_labeled.csv"):
    df = pd.read_csv(clean_csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.countplot(data=df, x="sentiment", order=["Positive", "Neutral", "Negative"], ax=axes[0])
    axes[0].set_title("Sentiment Distribution")
    
    sns.histplot(df["word_count"], bins=15, kde=True, ax=axes[1])
    axes[1].set_title("Comment Length Distribution")
    
    stopwords = {"là", "của", "và", "có", "cho", "ở", "thì", "mình", "các", "những", "này", "đó", "với", "được", "không"}
    all_words = [w for text in df["clean_text"].dropna() for w in text.split() if w not in stopwords and not w.isdigit()]
    top_words = Counter(all_words).most_common(15)
    if top_words:
        words, counts = zip(*top_words)
        sns.barplot(x=list(counts), y=list(words), ax=axes[2])
        axes[2].set_title("Top 15 Keywords")
    
    plt.tight_layout()
    plt.savefig("data/processed/eda_summary_week1.png", dpi=300)
    print("[OK] Đã lưu ảnh biểu đồ EDA tại: data/processed/eda_summary_week1.png")

if __name__ == "__main__":
    run_eda()
