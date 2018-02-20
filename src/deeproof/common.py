import logging
import re
import pandas as pd
from pathlib import Path

# Paths
PKG_PATH = Path(__file__).parent.parent.parent
SRC_PATH = PKG_PATH / 'src'
SUBMISSION_DIR = PKG_PATH / "submission"
PRETRAINED_DIR = PKG_PATH / "pretrained"
DATA_DIR = PKG_PATH / "data"
IMAGE_DIR = DATA_DIR / "images"
SNAPSHOT_DIR = PKG_PATH / "snapshots"

lst_dirs = [SUBMISSION_DIR, SUBMISSION_DIR, PRETRAINED_DIR,
            DATA_DIR, IMAGE_DIR, SNAPSHOT_DIR, ]
for p in lst_dirs:
    if not p.exists():
        p.mkdir()

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


# def get_images_ids():
#     """Get the IDs from the file names in the IMAGE_DIR directory."""
#     images_files = [f.as_posix() for f in IMAGE_DIR.iterdir()]
#     # Extract the image IDs from the file names. They will serve as an index later
#     image_ids = [int(re.sub(r'(^.*/|\.jpg)', '', fname))
#                  for fname in images_files]
#     return image_ids


# ALL_IDS = pd.Index(get_images_ids())

# DF_REFS = pd.read_csv(DATA_DIR / 'train.csv', index_col='id')
# IDS_LABELED = DF_REFS.index
# IDS_SUBMIT = ALL_IDS.difference(IDS_LABELED)


MAINLOG = "DeepRoof"


def setup_logs(save_dir, run_name):
    # initialize logger
    logger = logging.getLogger(MAINLOG)
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = save_dir / (run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
