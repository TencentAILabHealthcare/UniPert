__author__ = "Yiming Li"
__email__ = "liyiming5@qq.com"
__version__ = "1.0.0"

import os

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'current_model')
MMSEQS_CACHE_DIR = os.path.join(BASE_DIR, 'mmseqs_storage')

# The key may become invalid, currently only for testing
os.environ['CHEMSPIDER_APIKEY'] = 'PFQJTl2Ryn78O6fFpN9xH75oyXfVdZJS5TXX5UcS'

from .model import *