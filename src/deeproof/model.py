from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from deeproof.common import DATA_DIR, IMAGE_DIR, SNAPSHOT_DIR, SUBMISSION_DIR, setup_logs
from deeproof.dataset import RoofDataset, train_valid_split
from deeproof.train import train, snapshot
from deeproof.validation import validate
from deeproof.prediction import predict, write_submission_file


class DeepRoof():
    """Wrapper class."""
    def __init__(self, run_name, logger,
                 ds_transform_augmented, ds_transform_raw,
                 batch_size=4,
                 sampler=SubsetRandomSampler,
                 limit_load=None):

        self.run_name = run_name
        self.logger = logger

        self.ds_transform_augmented = ds_transform_augmented
        self.ds_transform_raw = ds_transform_raw

        # Loading the dataset
        X_train = RoofDataset(DATA_DIR / 'train.csv', IMAGE_DIR,
                              transform=ds_transform_augmented,
                              limit_load=limit_load,
                              )
        X_val = RoofDataset(DATA_DIR / 'train.csv', IMAGE_DIR,
                            transform=ds_transform_raw,
                            limit_load=limit_load,
                            )

        # Creating a validation split
        train_idx, valid_idx = train_valid_split(X_train, 0.2)

        if sampler is not None:
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = None, None

        # Both dataloader loads from the same dataset but with different indices
        train_loader = DataLoader(X_train,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=4,
                                  # pin_memory=True,
                                  )

        valid_loader = DataLoader(X_val,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=4,
                                  # pin_memory=True,
                                  )

        self.X_train, self.X_val = X_train, X_val
        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.train_loader, self.valid_loader = train_loader, valid_loader

    def train(self, epochs, model, loss_func, optimizer):
        best_score = 0.
        for epoch in range(epochs):
            epoch_timer = timer()

            # Train and validate
            train(epoch, self.train_loader, model, loss_func, optimizer)
            score, loss, threshold = validate(
                epoch, self.valid_loader, model, loss_func,
                self.X_train.getLabelEncoder())

            # Save
            is_best = score > best_score
            best_score = max(score, best_score)
            snapshot(SNAPSHOT_DIR, self.run_name, is_best, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'threshold': threshold,
                'val_loss': loss
            })

            end_epoch_timer = timer()
            self.logger.info("#### End epoch {}, elapsed time: {}".format(
                epoch, end_epoch_timer - epoch_timer))

        self.model = model

    def predict(self, batch_size=4, limit_load=None):
        X_test = RoofDataset(DATA_DIR / 'sample_submission.csv',
                             IMAGE_DIR,
                             transform=self.ds_transform_raw,
                             limit_load=limit_load,
                             )

        test_loader = DataLoader(X_test,
                                 batch_size=batch_size,
                                 num_workers=4,
                                 # pin_memory=True,
                                 )

        self.X_test, self.test_loader = X_test, test_loader

        # Load model from best iteration
        self.logger.info('===> loading best model for prediction')
        fpath = SNAPSHOT_DIR / (self.run_name + '-model_best.pth')
        checkpoint = torch.load(fpath.as_posix())
        self.model.load_state_dict(checkpoint['state_dict'])

        # Predict
        predictions = predict(test_loader, self.model)

        write_submission_file(predictions,
                              self.X_test.ids,
                              SUBMISSION_DIR,
                              self.run_name,
                              checkpoint['best_score'])
