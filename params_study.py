import logging
import pandas as pd
from common import TRAINED_DIR
from handle_data import SolarMapDatas
from model import SolarMapModel, LargeNet, ShortNet, generate_train_test_sets

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

    for s in range(40, 64, 6):
        resize = transforms.Resize((s, s))
        ratiocrop = int(224 / 225 * s)

        for m in range(2, 8):
            mean = [m * 0.1] * 3

            for st in range(2, 8):
                std = [st * 0.1] * 3

                normalize.mean = mean
                normalize.std = std

                logging.info('\n-------------------------------------\n'
                             'width = {}  ------   height = {}\n'
                             'mean = {}\n'
                             'std = {}'
                             .format(s, s, mean, std))

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
        trainset, testset,
        batch_size=4,
        num_workers=4,
    )

    lst_lossfunc = [
        nn.L1Loss(),
        nn.MSELoss(),
        nn.CrossEntropyLoss(),
        nn.NLLLoss2d(),
        nn.PoissonNLLLoss(),
        nn.BCELoss(),
    ]

    for i_learning_rate in range(1, 5):
        learning_rate = i_learning_rate * 0.001
        for criterion in [nn.CrossEntropyLoss(), nn.NLLLoss2d()]:
            # for optimizer in [optim.SGD, optim.RMSPro]:
            for optimizer in [optim.Adam]:
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
