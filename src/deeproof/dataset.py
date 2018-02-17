import pandas as pd
import PIL
import random
from math import floor

from sklearn.preprocessing import LabelBinarizer
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch import from_numpy, np


class RoofDataset(Dataset):
    """Dataset wrapping images and target labels for Engie Challenge.
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        optional: A torchvision transforms
    """

    def __init__(self, csv_path, img_path, img_ext='.jpg',
                 transform=transforms.Compose(
                     [transforms.Resize(size=(64, 64)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                           std=(0.5, 0.5, 0.5))]),
                 limit_load=None):

        df = pd.read_csv(csv_path, nrows=limit_load)
        ids_missing_mask = []
        for i, row in df.iterrows():
            fpath = img_path / (str(int(row['id'])) + img_ext)
            ids_missing_mask.append(fpath.exists())
        assert all(ids_missing_mask), \
            "Some images referenced in the CSV file where not found: {}".format(
                df['id'][[not i for i in ids_missing_mask]])

        df = df.set_index('id').sort_index()
        self.df = df

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.ids = self.df.index

        self.lb = LabelBinarizer()
        if 'orientation' in self.df.columns.tolist():
            self.labels = self.lb.fit_transform(self.df['orientation']).astype(np.float32)
        else:
            self.labels = np.zeros(self.df.shape)

    def __getitem__(self, index):
        """Return data at index."""
        fname = self.img_path / (str(self.ids[index]) + self.img_ext)
        img = PIL.Image.open(fname)
        if self.transform is not None:
            img = self.transform(img)

        label = from_numpy(self.labels[index])
        return img, label

    def __len__(self):
        return len(self.df.index)

    def getLabelEncoder(self):
        return self.lb


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
