from pathlib import Path
import os

__author__ = "Yiming Li"
__email__ = "liyiming5@qq.com"
__version__ = "0.1.0"


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "current_model"
MMSEQS_CACHE_DIR = BASE_DIR / "mmseqs_storage"


CHEMSPIDER_APIKEY = os.environ.get(
    "CHEMSPIDER_APIKEY",
    None
)

from .model import UniPert

__all__ = ["UniPert"]