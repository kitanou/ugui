import pytest

from ugui_core.models import Tweet
from ugui_core.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "ugui.sqlite3")


@pytest.fixture
def make_tweet():
    def make(id="1", text="冷たいそばが好き", day=1, **kwargs):
        return Tweet(id=id, text=text, timestamp=f"2025-01-{day:02d}T00:00:00+00:00", **kwargs)

    return make
