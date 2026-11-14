import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLES = RACINE / "samples"


@pytest.fixture(autouse=True)
def dossier_isole(tmp_path, monkeypatch):
    """Chaque test écrit ses réglages et son historique dans un dossier jetable."""
    monkeypatch.setenv("VOCALIS_DOSSIER", str(tmp_path / "vocalis"))
    return tmp_path / "vocalis"


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
