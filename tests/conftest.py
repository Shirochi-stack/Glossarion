import sys
import pathlib

# Ensure src/ is on sys.path for test imports
SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
