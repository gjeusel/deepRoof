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

# Custom imports
from deeproof.common import DATA_DIR, IMAGE_DIR, SNAPSHOT_DIR, SUBMISSION_DIR, setup_logs
from deeproof.neuro import ResNet50, ResNet101, ResNet152, ShortNet
from deeproof.dataset import RoofDataset, train_valid_split
from deeproof.model import DeepRoof


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
        transforms.RandomResizedCrop(224),
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    dr = DeepRoof(run_name, logger, ds_transform_augmented, ds_transform_raw)

    ##### Model parameters: #####
    # model = ResNet50(4)
    model = ShortNet((3, 224, 224))

    # criterion = ConvolutedLoss()
    weight = torch.Tensor([1., 1.971741, 3.972452, 1.824547])
    criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weight)

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
