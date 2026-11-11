# -*- coding: utf-8 -*-
"""Makes the flat src/ modules importable as `import core_config`, etc.
without requiring `pip install -e .` first."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
