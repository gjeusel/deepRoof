from collections import OrderedDict
import hashlib
import time
import logging
import torch
import pandas as pd
from deeproof.common import SNAPSHOT_DIR
from torchvision import transforms

from deeproof.common import MAINLOG
logger = logging.getLogger(MAINLOG)


class DataBaseModels():
    """Handle historic of already tested parameters."""

    id_model = None

    def __init__(self, store_dir=SNAPSHOT_DIR):
        self.store_dir = store_dir
        self.path = store_dir / 'references.csv'
        if not self.path.exists():
            df = pd.DataFrame(
                columns=['id', 'CNN_repr',
                         'epoch', 'best_score', 'loss', 'optimizer', 'criterion',
                         'transform_train', 'transform_test']
            )
            df.to_csv(self.path, index=False)
        self.df = pd.read_csv(self.path, index_col=0)

    # Private
    def _serialize_transform(self, transf):
        """Serialize a torchvision.transforms.Compose object."""
        assert isinstance(transf, transforms.Compose)

        t_json = []
        for t in transf.transforms:
            t_json.append({'type': str(t.__class__.__name__),  # only get transforms type
                           **vars(t),
                           })
        return str(t_json)

    def _serialize_optimizer(self, optimizer):
        opt_name = str(optimizer.__class__.__name__)
        opt_params = optimizer.defaults
        return opt_name + str(opt_params)

    def _construct_row_from_params(self, net, epoch, best_score, loss, optimizer,
                                   criterion, transform_train, transform_test):
        return [
            self._hashnet(net),
            epoch,
            best_score,
            loss,
            self._serialize_optimizer(optimizer),
            str(criterion.__class__.__name__),
            self._serialize_transform(transform_train),
            self._serialize_transform(transform_test),
        ]

    def _hashnet(self, net):
        net_name = net.__class__.__name__
        hashed = hashlib.sha224(str(net).encode('utf-8')).hexdigest()
        return net_name + ': ' + hashed

    # Public
    def model_exists(self, net, optimizer, criterion,
                     transform_train, transform_test):
        """Check if this model already exists."""

        self.df = pd.read_csv(self.path, index_col=0)

        bool_model = False
        for i, row in self.df.iterrows():
            bool_model = (
                row['CNN_repr'] == self._hashnet(net) and
                row['optimizer'] == self._serialize_optimizer(optimizer) and
                row['criterion'] == str(criterion.__class__.__name__) and
                row['transform_train'] == self._serialize_transform(transform_train) and
                row['transform_test'] == self._serialize_transform(
                    transform_test)
            )
            if bool_model:
                break

        return bool_model

    def get_id_model(self, net, optimizer, criterion,
                     transform_train, transform_test):
        """Return the id_model for the given parameters.
        If no existing were found, create a new id_model.
        """

        self.df = pd.read_csv(self.path, index_col=0)
        # First time
        if self.df.empty:
            return 0

        self.df = pd.read_csv(self.path, index_col=0)

        bool_model = False
        id_model = None
        for i, row in self.df.iterrows():
            bool_model = (
                row['CNN_repr'] == self._hashnet(net) and
                row['optimizer'] == self._serialize_optimizer(optimizer) and
                row['criterion'] == str(criterion.__class__.__name__) and
                row['transform_train'] == self._serialize_transform(transform_train) and
                row['transform_test'] == self._serialize_transform(
                    transform_test)
            )
            if bool_model:
                id_model = self.df.index[i]
                break

        if not bool_model:  # i.e this model doesn't exist yet
            id_model = self.df.index.max() + 1
            logging.info('New Model detected, id={}'.format(id_model))
        else:
            logging.info(
                'Already Trained model detected, id={}'.format(id_model))
        return id_model

    def save_model(self, net, epoch, best_score, loss, optimizer, criterion,
                   transform_train, transform_test):
        """Save CNN into .ckpt and update trained_models/references.csv file."""

        if self.id_model is None:
            self.id_model = self.get_id_model(net, optimizer, criterion,
                                              transform_train, transform_test)
        id_model = self.id_model

        self.df.loc[id_model] = self._construct_row_from_params(
            net, epoch, best_score, loss, optimizer, criterion,
            transform_train, transform_test)

        self.df.to_csv(self.path)

        snapshot_file = self.store_dir / \
            '{}_best_model.pth'.format(self.id_model)

        state = {
            'epoch': epoch,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': loss,
        }
        torch.save(state, snapshot_file.as_posix())
        logger.info("Snapshot saved to {}".format(snapshot_file.as_posix()))

    def get_existing_cnn(self, id_model):
        """Return what is needed to continue training."""
        snapshot_file = self.store_dir / \
            '{}_best_model.pth'.format(id_model)
        state_dict = torch.load(snapshot_file.as_posix())

        net_state_dict = state_dict['net']
        optimizer_state_dict = state_dict['optimizer']
        epoch = state_dict['epoch']

        return net_state_dict, optimizer_state_dict, epoch
