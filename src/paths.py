import os
from pathlib import Path

# allows to locate root dir of project easily
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
SRC_DIR = ROOT_DIR / "src"
TEST_DIR = ROOT_DIR / "test"
