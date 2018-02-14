import logging
import pandas as pd
from common import TRAINED_DIR
from handle_data import SolarMapDatas
from model import SolarMapModel, generate_train_test_sets
from networks import ShortNet, LargeNetSmallPictures, LargeNet

from sklearn.metrics import average_precision_score

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

normalize = transforms.Normalize(mean=[0.3, 0.3, 0.3],
                                 std=[0.2, 0.2, 0.2])

transform_train = transforms.Compose([
    transforms.Resize((46, 46)),
    transforms.RandomResizedCrop(45),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize((46, 46)),
    transforms.RandomResizedCrop(45),
    transforms.ToTensor(),
    normalize,
])


preproc_param = {
    'transform_train': transform_train,
    'transform_test': transform_test,
}

trainset, testset = generate_train_test_sets(
    mode='train-test', **preproc_param)

model = SolarMapModel(
    trainset,
    testset,
    batch_size=4,
    num_workers=4,
)

hyper_param = dict(
    num_epochs=15,
    criterion=nn.CrossEntropyLoss(weight=torch.Tensor([1., 1.971741, 3.972452, 1.824547])),
    optimizer_func=optim.Adam,
    optimizer_args={'lr': 0.001},
)
