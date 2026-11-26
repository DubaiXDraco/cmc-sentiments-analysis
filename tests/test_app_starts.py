from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app_demo.py")


def test_app_starts_and_predicts_example():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    # Bấm ví dụ nhanh đầu tiên (button[1]) rồi bấm "Phân tích ngay" (button[0])
    at.button[1].click().run()
    assert at.text_area[0].value.strip() != ""

    at.button[0].click().run()
    assert not at.exception
    results = [m.value for m in at.markdown if m.value.startswith("### Kết quả:")]
    assert len(results) == 1
    assert any(lbl in results[0] for lbl in ("Positive", "Neutral", "Negative"))


def test_app_empty_input_shows_warning():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    assert not [m for m in at.markdown if m.value.startswith("### Kết quả:")]
    assert len(at.warning) >= 1
