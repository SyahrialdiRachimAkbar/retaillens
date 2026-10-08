"""Smoke-test dashboard pages and filter behavior against generated real data."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
assert not app.exception, str(app.exception)
assert len(app.metric) == 4
print("Overview: passed")
for page in ["Customer segments", "Purchase predictions", "Model lab", "Data & methodology"]:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"{page}: {app.exception}"
    print(f"{page}: passed")
app.sidebar.radio[0].set_value("Overview").run()
global_sales = app.metric[0].value
app.sidebar.selectbox[0].set_value("France").run()
assert not app.exception
assert app.metric[0].value != global_sales
print("Market filter changes real sales totals: passed")
app.sidebar.radio[0].set_value("Purchase predictions").run()
assert not app.exception
app.slider[0].set_value(100).run()
assert not app.exception
assert app.metric[0].value == "0"
assert any("No customers match" in item.value for item in app.info)
print("Prediction threshold and empty state: passed")
