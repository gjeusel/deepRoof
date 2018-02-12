import logging
import pandas as pd
import numpy as np

from common import (IDS_LABELED, IDS_SUBMIT,
                    CLASSES, DF_REFS,
                    IMAGE_DIR,
                    )

import torch
import torchvision
from torchvision import transforms
import PIL

import matplotlib.pyplot as plt


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
    """load datas for list of ids."""
    images = []
    counter = 0
    for i in ids:
        fname = '{id}.jpg'.format(id=i)
        fpath = (IMAGE_DIR / fname).as_posix()
        im = PIL.Image.open(fpath)
        im.load()
        images.append(im)
        if (counter != 0) and ((counter % 1000) == 0):
            logging.info("{}th image loaded.".format(counter))
        counter += 1
    return images


class SolarMapDatas(torch.utils.data.Dataset):
    """Base class to handle datas for SolarMap Challenge."""
    # cf torchvision/torchvision/datasets/cifar.py for example

    ids = None
    mode = 'train-test'
    transform = None
    df_classe = None

    def __init__(self,
                 ids=IDS_LABELED,
                 transform=transforms.Compose(
                     [transforms.Resize(size=(64, 64)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                           std=(0.5, 0.5, 0.5)),
                      ]),
                 limit_load=None,
                 ):

        self.transform = transform

        ids = pd.Index(ids).sort_values()
        size = len(ids) if limit_load is None else limit_load
        self.ids = ids[:size]

        self.mode = guess_mode(ids)

        self.images = load_raw_images(self.ids)
        self.width, self.height = list(transform(self.images[0]).shape[1:])

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
        im = self.images[index]
        im = self.transform(im)

        if self.mode == 'train-test':
            return im, self.labels[index]
        else:
            return im

    def __len__(self):
        return len(self.images)

    def _check_integrity(self):
        # TODO
        pass

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Width, Height: {},{}\n'.format(self.width, self.height)
        fmt_str += '    Split: {}\n'.format(self.mode)
        fmt_str += '    Image directory: {}\n'.format(IMAGE_DIR)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
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
