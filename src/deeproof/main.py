# Utilities
import random
import logging
import time
from timeit import default_timer as timer

# Libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Custom imports
from deeproof.common import DATA_DIR, IMAGE_DIR, SNAPSHOT_DIR, SUBMISSION_DIR, setup_logs
from deeproof.neuro import ResNet50, ResNet101, ResNet152
from deeproof.dataset import RoofDataset, train_valid_split
from deeproof.train import train, snapshot
from deeproof.validation import validate
from deeproof.prediction import predict, write_submission_file


class DeepRoof():
    """Wrapper class."""
    def __init__(self, run_name, logger,
                 ds_transform_augmented, ds_transform_raw,
                 batch_size=4):

        self.run_name = run_name
        self.logger = logger

        # Loading the dataset
        X_train = RoofDataset(DATA_DIR / 'train.csv', IMAGE_DIR,
                              transform=ds_transform_augmented,
                              )
        X_val = RoofDataset(DATA_DIR / 'train.csv', IMAGE_DIR,
                            transform=ds_transform_raw,
                            )

        # Creating a validation split
        train_idx, valid_idx = train_valid_split(X_train, 0.2)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

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

    def predict(self):
        X_test = RoofDataset(DATA_DIR / 'sample_submission.csv',
                             IMAGE_DIR,
                             transform=self.ds_transform_raw,
                             )

        test_loader = DataLoader(X_test,
                                 batch_size=4,
                                 num_workers=4,
                                 # pin_memory=True,
                                 )

        self.X_test, self.test_loader = X_test, test_loader

        # Load model from best iteration
        self.logger.info('===> loading best model for prediction')
        checkpoint = torch.load(
            SNAPSHOT_DIR, self.run_name + '-model_best.pth')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Predict
        predictions = predict(test_loader, self.model)

        write_submission_file(predictions,
                              checkpoint['threshold'],
                              self.X_test,
                              self.X_train.getLabelEncoder(),
                              SUBMISSION_DIR,
                              self.run_name,
                              checkpoint['best_score'])


if __name__ == "__main__":
    # Initiate timer:
    global_timer = timer()

    # Setup logs
    run_name = time.strftime("%Y-%m-%d_%H%M-") + "resnet50-L2reg-new-data"
    logger = setup_logs(SNAPSHOT_DIR, run_name)

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    # torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    ##### Preprocessing parameters: #####

    # Normalization on ImageNet mean/std for finetuning
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Augmentation + Normalization for full training
    ds_transform_augmented = transforms.Compose([
        transforms.RandomSizedCrop(224),
        # PowerPIL(),
        transforms.ToTensor(),
        # ColorJitter(), # Use PowerPIL instead, with PillowSIMD it's much more efficient
        normalize,
        # Affine(
        #    rotation_range = 15,
        #    translation_range = (0.2,0.2),
        #    shear_range = math.pi/6,
        #    zoom_range=(0.7,1.4)
        # )
    ])

    # Normalization only for validation and test
    ds_transform_raw = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        normalize
    ])

    dr = DeepRoof(run_name, logger, ds_transform_augmented, ds_transform_raw)

    ##### Model parameters: #####
    model = ResNet50(4)

    # criterion = ConvolutedLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss(
        weight=torch.Tensor([1,  4,  2,  1,
                             1,  3,  3,  3,
                             4,  4,  1,  2,
                             1,  1,  3,  4,  1])
    )

    # Note, p_training has lr_decay automated
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9,
                          weight_decay=0.0005)  # Finetuning whole model

    # Training:
    dr.train(epochs=16, model=model, loss_func=criterion, optimizer=optimizer)

    # Predict:
    dr.predict()

    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
