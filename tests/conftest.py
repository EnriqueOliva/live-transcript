import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def settings_path(tmp_path):
    return tmp_path / "config" / "settings.json"
