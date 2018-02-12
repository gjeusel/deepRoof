import logging
import pandas as pd
from common import TRAINED_DIR
from handle_data import SolarMapDatas
from model import SolarMapModel, LargeNet, ShortNet, generate_train_test_sets
from historic_models import serialize_transform

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


def explore_preproc_params_ShortNet():

    # Set of parameters inspired from :
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    hyper_param = dict(
        num_epochs=5,
        criterion=nn.CrossEntropyLoss(),
        optimizer_func=optim.Adam,
        optimizer_args={'lr': 0.001},
    )

    for s in range(64, 96, 8):
        resize = transforms.Resize((s, s))
        ratiocrop = int(224 / 225 * s)

        logging.info('\n-------------------------------------\n'
                     'width = {}  ------   height = {}\n' .format(s, s))

        for horiz in [transforms.RandomVerticalFlip(),
                      transforms.RandomHorizontalFlip()]:
            preproc_param = dict(
                transform_train=transforms.Compose(
                    [resize,
                     transforms.RandomResizedCrop(size=ratiocrop),
                     horiz,
                     transforms.ToTensor(),
                     normalize,
                     ]),
                transform_test=transforms.Compose(
                    [resize,
                     transforms.CenterCrop(size=ratiocrop),
                     transforms.ToTensor(),
                     normalize,
                     ]),
            )

            trainset, testset = generate_train_test_sets(
                mode='train-test', **preproc_param)

            model = SolarMapModel(
                trainset, testset,
                batch_size=4,
                num_workers=4,
            )

            model.process(CNN_type=ShortNet, **hyper_param)


def explore_hyperparams_ShortNet():
    preproc_param = dict(
        transform_train=transforms.Compose(
            [transforms.Resize(size=(96, 96)),
                transforms.RandomResizedCrop(size=88),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
             ]),
        transform_test=transforms.Compose(
            [transforms.Resize(size=(96, 96)),
             transforms.CenterCrop(size=(88, 88)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
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
            for optimizer in [optim.SGD, optim.RMSPro]:
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
