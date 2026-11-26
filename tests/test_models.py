import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import LABELS, SentimentEngine


def _check_engine_prediction(engine: SentimentEngine, expected_backend: str):
    assert engine.backend == expected_backend
    res = engine.predict("Phòng lab máy cấu hình cao học AI mượt cực")
    assert res.label in LABELS
    assert res.label_vi in ("Tiêu cực", "Bình thường", "Tích cực")
    assert 0.0 <= res.confidence <= 1.0
    assert math.isclose(sum(res.probabilities.values()), 1.0, abs_tol=1e-2)
    assert res.latency_ms >= 0.0


def test_tfidf_lr_predict_and_explainability():
    engine = SentimentEngine(backend="tfidf_lr")
    _check_engine_prediction(engine, "tfidf_lr")
    res = engine.predict("Wifi trường yếu quá, đăng ký môn học lag suốt")
    assert len(res.top_features) > 0, "TF-IDF + LR phải trả về top features để giải thích"


def test_tfidf_svm_predict():
    engine = SentimentEngine(backend="tfidf_svm")
    _check_engine_prediction(engine, "tfidf_svm")


def test_phobert_local_weights_load_and_predict():
    engine = SentimentEngine(backend="phobert")
    _check_engine_prediction(engine, "phobert")
    batch = engine.predict_batch(
        [
            "Cơ sở vật chất mới đẹp và hiện đại",
            "Cho mình hỏi lịch thi cuối kỳ xem ở đâu vậy?",
            "Đăng ký môn học server sập suốt, chán quá",
        ]
    )
    assert len(batch) == 3
    for item in batch:
        assert item.label in LABELS
        assert math.isclose(sum(item.probabilities.values()), 1.0, abs_tol=1e-2)
