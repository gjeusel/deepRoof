#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
from pathlib import Path

import random

import pandas as pd
import numpy as np

import torch
import torchvision
from torchvision import transforms
import PIL


import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data/"
IMAGE_DIR = DATA_DIR / "images/"

DF_TRAIN = pd.read_csv(DATA_DIR / 'train.csv', index_col='id')

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


IDS = pd.Index(get_images_ids())

IDS_TRAIN = DF_TRAIN.index
IDS_SUBMIT = IDS.difference(IDS_TRAIN)

WIDTH, HEIGHT = 96, 96

# transform = torchvision.transforms.Compose(
#     [torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )


class SolarMapDatas(torch.utils.data.Dataset):
    """Base class to handle datas for SolarMap Challenge."""

    base_folder = 'images'
    # cf torchvision/torchvision/datasets/cifar.py for example

    def __init__(self, root='./data',
                 lst_ids=IDS_TRAIN,
                 limit_load=None,
                 transform=None,
                 download=False,
                 verbose=True):

        self.root = os.path.expanduser(root)
        self.transform = transform

        self.mode = self.guess_mode(lst_ids)

        self.lst_ids = lst_ids

        self.verbose = verbose
        self.limit_load = np.inf if limit_load is None else limit_load

        if download:
            # TODO
            pass

        if self.mode == 'train-test':
            self.images, np_data, labels = \
                self.load_raw_images(self.lst_ids)
        elif self.mode == 'submit':
            self.images, self.np_data, _ = \
                self.load_raw_images(self.lst_ids)

    def guess_mode(self, lst_ids):
        """Guess the purpose of instanciated class based on lst_ids given"""
        if np.isin(IDS_TRAIN, lst_ids).all():
            mode = 'train-test'
        elif np.isin(IDS_SUBMIT, lst_ids).all():
            mode = 'submit'
        else:
            assert False

        return mode

    def load_raw_images(self, lst_ids):
        """Load datas for lst_ids."""
        images = {}
        images_asarray = []
        labels = []

        counter = 0
        for i in lst_ids:
            f = '{id}.jpg'.format(id=i)
            fname = os.path.join(self.root, self.base_folder, f)
            im = PIL.Image.open(fname)
            im.load()

            images[i] = im

            im = im.resize((WIDTH, HEIGHT), resample=PIL.Image.ANTIALIAS)
            images_asarray.append(np.asarray(im))

            if self.mode == 'train-test':
                labels.append(DF_TRAIN.loc[i])

            if self.verbose and (counter != 0) and ((counter % 1000) == 0):
                print("{}th image loaded.".format(counter))

            counter += 1
            if counter > self.limit_load - 1:
                break

        num_images = min(len(lst_ids), self.limit_load)

        images_asarray = np.concatenate(images_asarray)
        images_asarray = images_asarray.reshape((num_images, WIDTH, HEIGHT, 3))
        labels = np.array(labels)
        return images, images_asarray, labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where image is a torch.FloatTensor and
                   target is index of the target class.
        """

        img = self.np_data[index]

        if self.transform is not None:
            img = self.transform(img)

        # Always assure that tensor is returned:
        transf = transforms.Compose([transforms.ToTensor()])
        img = transf(img)

        if self.mode == 'train-test':
            target = self.labels[index]
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.np_data)

    def _check_integrity(self):
        # TODO
        pass

    def download(self):
        # TODO
        pass

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
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
        # img = img / 2 + 0.5  # unnormalize
        # npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        transf = transforms.Compose([transforms.ToPILImage()])
        PIL_img = transf(img)
        plt.imshow(PIL_img)

    def plot_images(self, **kwargs):
        """cf torchvision.utils.make_grid for args."""
        # Get some random training images
        dataiter = iter(self.loader)
        images, labels = dataiter.next()

        # Show images
        classes = []
        for i in range(len(labels)):
            classes.append(CLASSES[labels[i][0]])

        plt.title(' ; '.join(tuple(classes)))
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
