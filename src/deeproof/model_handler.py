from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from deeproof.common import DATA_DIR, IMAGE_DIR, SNAPSHOT_DIR, SUBMISSION_DIR, setup_logs
from deeproof.dataset import RoofDataset, train_valid_split
from deeproof.train import train
from deeproof.validation import validate
from deeproof.prediction import predict, write_submission_file
from deeproof.database_models import DataBaseModels


class DeepRoofHandler():
    """Wrapper class."""

    def __init__(self, logger,
                 ds_transform_augmented, ds_transform_raw,
                 batch_size=4,
                 num_workers=4,
                 sampler=SubsetRandomSampler,
                 limit_load=None):

        self.logger = logger
        self.dbmodel = DataBaseModels()

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
        train_idx = range(len(X_train))

        if sampler is not None:
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = None, None

        # Both dataloader loads from the same dataset but with different indices
        train_loader = DataLoader(X_train,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  # pin_memory=True,
                                  )

        valid_loader = DataLoader(X_val,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers,
                                  # pin_memory=True,
                                  )

        self.X_train, self.X_val = X_train, X_val
        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.train_loader, self.valid_loader = train_loader, valid_loader

    def train(self, epochs, model, loss_func, optimizer,
              resume_training_if_exists=True,
              record_model=True):
        """Train or continue training."""

        if resume_training_if_exists:
            model_exists = self.dbmodel.model_exists(
                model, optimizer, loss_func,
                self.ds_transform_augmented,
                self.ds_transform_raw)
            if model_exists:
                id_model = self.dbmodel.get_id_model(model, optimizer, loss_func,
                                                     self.ds_transform_augmented,
                                                     self.ds_transform_raw)
                model_state_dict, optimizer_state_dict, epoch_start = \
                    self.dbmodel.get_existing_cnn(id_model)
                model.load_state_dict(model_state_dict)
                optimizer.load_state_dict(optimizer_state_dict)
            else:
                epoch_start = 0

        best_score = 0.
        for epoch in range(epoch_start, epochs):
            epoch_timer = timer()

            # Train and validate
            train(epoch, self.train_loader, model, loss_func, optimizer)
            score, loss = validate(epoch, self.valid_loader, model, loss_func)

            # Save
            is_best = score > best_score
            best_score = max(score, best_score)

            if is_best and record_model:
                self.dbmodel.save_model(model, epoch+1, best_score, loss,
                                        optimizer, loss_func,
                                        self.ds_transform_augmented,
                                        self.ds_transform_raw)

            end_epoch_timer = timer()
            self.logger.info("#### End epoch {}, elapsed time: {}".format(
                epoch, end_epoch_timer - epoch_timer))

        self.model = model

    def predict(self, id_model, model, batch_size=4, num_workers=4, limit_load=None):
        X_test = RoofDataset(DATA_DIR / 'sample_submission.csv',
                             IMAGE_DIR,
                             transform=self.ds_transform_raw,
                             limit_load=limit_load,
                             )

        test_loader = DataLoader(X_test,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 # pin_memory=True,
                                 )

        self.X_test, self.test_loader = X_test, test_loader

        # Load model from best iteration
        self.logger.info('===> loading best model for prediction')
        model_state_dict, optimizer_state_dict, epoch_start = \
            self.dbmodel.get_existing_cnn(id_model)
        model.load_state_dict(model_state_dict)

        # Predict
        predictions = predict(test_loader, model)

        write_submission_file(predictions,
                              self.X_test.ids,
                              SUBMISSION_DIR,
                              id_model,
                              self.dbmodel.df.loc[0]['best_score'],
                              )
