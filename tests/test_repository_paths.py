from pathlib import Path

from paths import get_data_path, get_repository_path, get_visuals_path


def test_repository_paths_use_configured_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ALBUS_PATH", str(tmp_path))
    assert get_repository_path() == tmp_path
    assert get_data_path() == tmp_path / "data"
    assert get_visuals_path() == tmp_path / "visuals"


def test_repository_path_defaults_to_checkout(monkeypatch):
    monkeypatch.delenv("ALBUS_PATH", raising=False)
    assert get_repository_path() == Path(__file__).resolve().parents[1]