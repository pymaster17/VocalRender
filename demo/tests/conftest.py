"""Make demo/app.py importable and keep model loading out of unit tests."""
import os
import sys
from pathlib import Path

os.environ.setdefault("VOCALRENDER_UI_ONLY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
