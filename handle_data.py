#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data/"
IMAGE_DIR = DATA_DIR / "images/"

IMAGE_FILES = [f.as_posix() for f in IMAGE_DIR.iterdir()]
N = len(IMAGE_FILES)

ROOF_NORTH_SOUTH = 1
ROOF_WEST_EAST = 2
ROOF_FLAT = 3
ROOF_UNKNOWN = 4


def get_images_ids():
    """Get the IDs from the file names in the IMAGE_DIR directory."""
    # Extract the image IDs from the file names. They will serve as an index later
    image_ids = [int(re.sub(r'(^.*/|\.jpg)', '', fname)) for fname in IMAGE_FILES]
    return image_ids


def load_raw_images(n_limit=None):
    """Load the raw images.

    :n_limit: integer, limit of images to load. If None, load all.
    :returns: dict of <PIL.JpegImageFile>
    """
    images = {}
    image_ids = get_images_ids()
    for i, fname in enumerate(IMAGE_FILES):
        # Stop if n_limit defined:
        if n_limit is not None and i >= n_limit:
            break

        # Verbosity:
        if (i != 0) and not (i % 1000):
            print("Loading {} th image".format(i))

        # Get the ID of the image
        iid = image_ids[i]
        # Raw image
        image = Image.open(fname)
        image.load()
        images[iid] = image

    return images


def display_images(images, ids_display=None):
    """Display images with matplotlib for which id is in ids_display."""
    # Getting number of subplots:
    if ids_display is None:
        rows, cols = 2, 3
        # Selecting randomly rows*cols ids that exists in images:
        ids_display = [list(images.keys())[np.random.randint(0, rows * cols)]
                       for i in range(rows * cols)]
    else:
        quotient, remainder = divmod(len(ids_display), int(np.sqrt(len(ids_display))))
        rows = quotient
        cols = quotient + remainder

    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    samples = []
    i = 0
    while i < len(ids_display):
        ax = axes[int(i / cols), i % cols]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.imshow(images[ids_display[i]])
        samples.append(ids_display[i])
        i += 1

    return fig, axes


def plot_images_sizes_distribution(images):
    """Plot the average image size distribution heigths, widths.

    :images: dict of <PIL.JpegImageFile>
    :returns: fig matplotlib objects.
    """

    widths = [im.size[0] for im in images.values()]
    heights = [im.size[1] for im in images.values()]

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, s=0.5)
    plt.axis('equal')
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.title('Image sizes')
    plt.xlabel('Width')
    plt.ylabel('Height')

    # Average image size (width, height)
    sum(widths) / N, sum(heights) / N

    return fig
