import logging
import pandas as pd
from common import TRAINED_DIR
from handle_data import SolarMapDatas
from model import SolarMapModel, LargeNet, ShortNet, generate_train_test_sets

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


def test_ShortNet():
    preproc_param = dict(
        width=96,
        height=96,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ]),
        limit_load=None,
    )
    trainset, testset = generate_train_test_sets(
        mode='train-test', **preproc_param)
    model = SolarMapModel(
        trainset, testset,
        batch_size=4,
        num_workers=4,
    )
    for i_learning_rate in range(1, 5):
        learning_rate = i_learning_rate * 0.001
        for criterion in [nn.CrossEntropyLoss(), nn.NLLLoss2d()]:
            for optimizer in [optim.Adam, optim.SGD]:
                logging.info(
                    '\n-------------------------------------\n'
                    'learning_rate: {learning_rate}\n'
                    'criterion: {criterion}\n'
                    'optimizer: {optimizer}\n'
                    '\n-------------------------------------\n'
                    .format(learning_rate=learning_rate,
                            criterion=str(criterion),
                            optimizer=str(optimizer)))

                hyper_param = {
                    'num_epochs': 5,
                    'learning_rate': learning_rate,
                    'criterion': criterion,
                    'optimizer': optimizer,
                }

                model.process(CNN_type=ShortNet, **hyper_param)
