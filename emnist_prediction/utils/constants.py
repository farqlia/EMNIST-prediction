import os
from pathlib import Path

import numpy as np

CLASS_LABELS = np.array([chr(label + ord('A')) for label in range(26)])
IMG_SIZE = (28, 28)
PROJECT_ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT_DIR / 'data'
INPUT_DATA_DIR = DATA_DIR / 'input_data'
SUBMISSIONS_DIR = DATA_DIR / 'submissions'

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

SUBDATA_DIR = INPUT_DATA_DIR / 'subdata'

os.makedirs(SUBDATA_DIR, exist_ok=True)
