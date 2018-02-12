from collections import OrderedDict
import logging
import torch
import pandas as pd
from common import TRAINED_DIR
from torchvision import transforms


def serialize_transform(transf):
    """Serialize a torchvision.transforms.Compose object."""
    assert isinstance(transf, transforms.Compose)

    t_json = []
    for t in transf.transforms:
        if isinstance(t, transforms.ToTensor):
            t_json.append({'type': 'ToTensor'})
        elif isinstance(t, transforms.Resize):
            t_json.append({
                'type': 'Resize',
                'size': t.size,
                'interpolation': t.interpolation,
            })
        elif isinstance(t, transforms.Normalize):
            t_json.append({
                'type': 'Normalize',
                'mean': t.mean,
                'std': t.std,
            })
        elif isinstance(t, transforms.RandomResizedCrop):
            t_json.append({
                'type': 'RandomResizedCrop',
                'size': t.size,
                'scale': t.scale,
                'ratio': t.ratio,
                'interpolation': t.interpolation,
            })

        else:
            logging.error('Implement serialization for {} plz !'
                          .format(type(t)))
    return t_json


class HistoricModel():
    """Handle historic of already tested parameters."""

    def __init__(self):
        self.path = TRAINED_DIR / 'references.csv'
        if not self.path.exists():
            df = pd.DataFrame(
                columns=['id', 'CNN_repr', 'hyper_param', 'width', 'height',
                         'transform_train', 'transform_test',
                         'train_progress(%)', 'accuracy', 'score']
            )
            df.to_csv(self.path, index=False)
        self.df = pd.read_csv(self.path, index_col=0)

    def inspect_models(self, CNN_repr, hyper_param, transform_train, transform_test):
        """Check if this model already exists."""
        self.df = pd.read_csv(self.path, index_col=0)
        for i, row in self.df.iterrows():
            model_already_exists = (
                str(CNN_repr) == row['CNN_repr']
                and str(OrderedDict(sorted(hyper_param.items()))) == row['hyper_param']
                and str(serialize_transform(transform_train)) == row['transform_train']
                and str(serialize_transform(transform_test)) == row['transform_test']
            )
            if model_already_exists:
                return self.df.index[i]
        return None  # otherwise

    def get_id_model(self, CNN_repr, hyper_param, transform_train, transform_test):
        """Return the id_model for the given parameters.
        If no existing were found, create a new id_model.
        """

        self.df = pd.read_csv(self.path, index_col=0)
        # First time
        if self.df.empty:
            return 0

        id_model = self.inspect_models(
            CNN_repr, hyper_param, transform_train, transform_test
        )
        if id_model is None:  # i.e this model doesn't exist yet
            id_model = self.df.index.max() + 1
            logging.info('New Model detected, id={}'.format(id_model))
        else:
            logging.info(
                'Already Trained model detected, id={}'.format(id_model))
        return id_model

    def save_model(self, id_model,
                   net, optimizer, epoch_computed,
                   hyper_param, transform_train, transform_test,
                   width, height,
                   train_progress, accuracy=None, score=None):
        """Save CNN into .ckpt and update trained_models/references.csv file."""

        logging.info(
            'Training progress: {}%  --- Saving Model.'.format(train_progress))
        self.df.loc[id_model] = [str(net),
                                 OrderedDict(sorted(hyper_param.items())),
                                 width, height,
                                 serialize_transform(transform_train),
                                 serialize_transform(transform_test),
                                 train_progress, accuracy, score]
        self.df.to_csv(self.path)

        ckpt_fname = 'model_{}.ckpt'.format(id_model)
        ckpt_path = (TRAINED_DIR / ckpt_fname).as_posix()
        torch.save({
            'epoch': epoch_computed,
            'net': net.state_dict(),
            'optimizer': optimizer},
            ckpt_path
        )

    def add_score(self, id_model, accuracy, score):
        self.df = pd.read_csv(self.path, index_col=0)
        self.df.loc[id_model, 'accuracy'] = accuracy
        self.df.loc[id_model, 'score'] = score
        self.df.to_csv(self.path)

    def get_existing_cnn(self, id_model):
        """Return what is needed to continue training."""
        ckpt_fname = 'model_{}.ckpt'.format(id_model)
        ckpt_path = (TRAINED_DIR / ckpt_fname).as_posix()
        state_dict = torch.load(ckpt_path)

        new_state_dict = state_dict['net']
        optimizer = state_dict['optimizer']
        epoch = state_dict['epoch']

        return new_state_dict, optimizer, epoch
