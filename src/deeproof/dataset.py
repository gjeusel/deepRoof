import pandas as pd
import PIL
import random
from math import floor

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch import from_numpy


class RoofDataset(Dataset):
    """Dataset wrapping images and target labels for Engie Challenge.
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        optional: A torchvision transforms
    """

    def __init__(self, csv_path, img_path, img_ext='jpg',
                 transform=transforms.Compose(
                     [transforms.Resize(size=(64, 64)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                           std=(0.5, 0.5, 0.5))])):

        self.df = pd.read_csv(csv_path, index_col=0)
        self.df = self.df.sort_index()
        self.df['orientation'] = self.df['orientation'] - \
            1  # begin class id at 0
        assert self.df.index.apply(
            lambda x: (img_path / x + img_ext).exists()).all(), \
            "Some images referenced in the CSV file were not found"

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.ids = self.df.index
        self.labels = self.df['orientation']

    def __getitem__(self, index):
        """Return data at index."""
        img = PIL.Image.open(self.img_path / self.ids[index] + self.img_ext)
        if self.transform is not None:
            img = self.transform(img)

        label = from_numpy(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.df.index)


def train_valid_split(dataset, test_size=0.25, shuffle=False, random_seed=0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.

    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = len(dataset)
    indices = list(range(1, length))

    if shuffle:
        random.seed(random_seed)
        random.shuffle(indices)

    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]
