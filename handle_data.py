#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import random

import pandas as pd
import numpy as np

import torch
import torchvision
from torchvision import transforms
import PIL

import matplotlib.pyplot as plt

SUBMISSION_DIR = Path(__file__).parent / "submission/"
DATA_DIR = Path(__file__).parent / "data/"
IMAGE_DIR = DATA_DIR / "images/"

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

# WIDTH, HEIGHT = 96, 96
WIDTH, HEIGHT = 32, 32


def guess_mode(ids):
    """Guess the purpose of instanciated class based on ids given"""
    if pd.Index(ids).isin(IDS_LABELED).all():
        mode = 'train-test'
    elif pd.Index(ids).isin(IDS_SUBMIT).all():
        mode = 'submit'
    else:
        assert False
    return mode


def load_raw_images(ids):
    """Load datas for list of ids."""
    images = {}
    images_asarray = []

    counter = 0
    for i in ids:
        f = '{id}.jpg'.format(id=i)
        fname = (IMAGE_DIR / f).as_posix()
        im = PIL.Image.open(fname)
        im.load()

        images[i] = im

        im = im.resize((WIDTH, HEIGHT), resample=PIL.Image.ANTIALIAS)
        images_asarray.append(np.asarray(im))

        if (counter != 0) and ((counter % 1000) == 0):
            print("{}th image loaded.".format(counter))
        counter += 1

    images_asarray = np.concatenate(images_asarray)
    images_asarray = images_asarray.reshape((len(ids), WIDTH, HEIGHT, 3))
    return images, images_asarray


class SolarMapDatas(torch.utils.data.Dataset):
    """Base class to handle datas for SolarMap Challenge."""
    # cf torchvision/torchvision/datasets/cifar.py for example

    ids = None
    mode = 'train-test'
    transform = None
    df_classe = None

    def __init__(self,
                 ids=IDS_LABELED,
                 limit_load=None,
                 transform=None,
                 ):

        self.transform = transform

        ids = pd.Index(ids).sort_values()
        size = len(ids) if limit_load is None else limit_load
        self.ids = ids[:size]

        self.mode = guess_mode(ids)

        self.images, self.np_data = load_raw_images(self.ids)

        if self.mode == 'train-test':
            self.labels = []
            self.df_classe = pd.DataFrame(
                0., index=self.ids, columns=list(CLASSES.values()))
            for i in self.ids:
                # Simple label
                orientation = DF_REFS.loc[i]['orientation']
                self.labels.append(orientation)

                # Cosmetic Dataframe:
                self.df_classe.loc[i][CLASSES[orientation]] = 1.

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the label class.
        """
        if self.mode == 'train-test':
            img, label = self.np_data[index], self.labels[index]
        else:
            img = self.np_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        else:
            transf = transforms.Compose([transforms.ToTensor()])
            img = transf(img)

        if self.mode == 'train-test':
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.np_data)

    def _check_integrity(self):
        # TODO
        pass

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        fmt_str += '    Image directory: {}\n'.format(IMAGE_DIR)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        return fmt_str


class SolarMapVisu(SolarMapDatas):
    """Base class to get plots done."""

    def __init__(self, batch_size=4, shuffle=True, num_workers=2,
                 *a, **kwargs):
        super().__init__(*a, **kwargs)
        self.loader = torch.utils.data.DataLoader(self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)

    def imshow(self, img):
        transf = transforms.Compose([transforms.ToPILImage()])
        PIL_img = transf(img)
        plt.imshow(PIL_img)

    def plot_images(self, **kwargs):
        """cf torchvision.utils.make_grid for args."""
        # Get some random training images
        dataiter = iter(self.loader)

        if self.mode == 'train-test':
            images, labels = dataiter.next()
            # Show images
            classes = []
            for i in range(len(labels)):
                classes.append(CLASSES[labels[i]])
            plt.title(' ; '.join(tuple(classes)))

        else:
            images = dataiter.next()

        self.imshow(torchvision.utils.make_grid(images, **kwargs))

    def plot_images_sizes_distrib(self):
        widths = [im.size[0] for im in self.images.values()]
        heights = [im.size[1] for im in self.images.values()]

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(widths, heights, s=0.5)
        plt.axis('equal')
        plt.xlim([0, 200])
        plt.ylim([0, 200])
        plt.title('Image sizes')
        plt.xlabel('Width')
        plt.ylabel('Height')

        return fig
