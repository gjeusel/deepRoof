import logging
from collections import OrderedDict
import torch
import re
import pandas as pd
from pathlib import Path

SUBMISSION_DIR = Path(__file__).parent / "submission/"
DATA_DIR = Path(__file__).parent / "data/"
IMAGE_DIR = DATA_DIR / "images/"
TRAINED_DIR = Path(__file__).parent / "trained_models/"

for p in [SUBMISSION_DIR, DATA_DIR, IMAGE_DIR, TRAINED_DIR]:
    if not p.exists():
        p.mkdir()

DF_REFS = pd.read_csv(DATA_DIR / 'train.csv', index_col='id')

ROOF_NORTH_SOUTH = 1
ROOF_WEST_EAST = 2
ROOF_FLAT = 3
ROOF_UNKNOWN = 4

CLASSES = {
    ROOF_NORTH_SOUTH: 'North-South',
    ROOF_WEST_EAST: 'West-East',
    ROOF_FLAT: 'Flat',
    ROOF_UNKNOWN: 'Unknown',
}


def get_images_ids():
    """Get the IDs from the file names in the IMAGE_DIR directory."""
    images_files = [f.as_posix() for f in IMAGE_DIR.iterdir()]
    # Extract the image IDs from the file names. They will serve as an index later
    image_ids = [int(re.sub(r'(^.*/|\.jpg)', '', fname))
                 for fname in images_files]
    return image_ids


ALL_IDS = pd.Index(get_images_ids())

IDS_LABELED = DF_REFS.index
IDS_SUBMIT = ALL_IDS.difference(IDS_LABELED)
