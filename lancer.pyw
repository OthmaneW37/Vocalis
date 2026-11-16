"""Lance Vocalis sans fenêtre de console (double-clic, ou démarrage de Windows)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vocalis.app import main  # noqa: E402

sys.exit(main())
