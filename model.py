#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime

from handle_data import (SolarMapDatas,
                         IDS_LABELED, IDS_SUBMIT, ALL_IDS, CLASSES,
                         SUBMISSION_DIR,
                         )

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim

from sklearn.metrics import average_precision_score

# from keras.applications import VGG16
# from keras.optimizers import SGD

# Let's get inspired from http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


def get_conv_output(shape, layer):
    bs = 1
    x = Variable(torch.rand(bs, *shape))
    out = layer(x)
    return out.size()[1:]


class LargeNet(nn.Module):
    """Convolutional Neural Network deep layers."""

    def __init__(self, input_shape):
        """
        :param tuple input_shape: shape of inputs tensor.
            example: (3, 96, 96) with 3 for RGB.
        """
        super(LargeNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 110, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(110, 84, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(84, 84, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(84, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        shape = get_conv_output(input_shape, self.layer1)
        shape = get_conv_output(shape, self.layer2)
        shape = get_conv_output(shape, self.layer3)
        shape = get_conv_output(shape, self.layer4)

        self.fc1 = nn.Linear(shape[0] * shape[1] *
                             shape[2], 120)  # fully connected
        self.fc2 = nn.Linear(120, 84)  # fully connected
        self.fc3 = nn.Linear(84, 4)  # fully connected

    def forward(self, x, training=True):
        out = self.layer1(x)
        out = nn.functional.dropout2d(out, 0.15)
        out = self.layer2(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = self.layer3(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = self.layer4(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class ShortNet(nn.Module):
    """Convolutional Neural Network short layers."""

    def __init__(self, input_shape):
        """
        :param tuple input_shape: shape of inputs tensor.
            example: (3, 96, 96) with 3 for RGB.
        """
        super(ShortNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        shape = get_conv_output(input_shape, self.layer1)
        shape = get_conv_output(shape, self.layer2)

        self.fc1 = nn.Linear(shape[0] * shape[1] *
                             shape[2], 120)  # fully connected
        self.fc2 = nn.Linear(120, 84)  # fully connected
        self.fc3 = nn.Linear(84, 4)  # fully connected

    def forward(self, x, training=True):
        out = self.layer1(x)
        out = nn.functional.dropout2d(out, 0.15)
        out = self.layer2(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def generate_train_test_sets(mode, perc=0.7, **kwargs):
    if mode == 'train-test':
        percentage_train = perc
        ids_train = random.sample(
            set(IDS_LABELED), int(len(IDS_LABELED) * percentage_train))
        ids_train = pd.Index(ids_train)
        ids_test = IDS_LABELED.difference(ids_train)

        ids_train = ids_train
        ids_test = ids_test
    elif mode == 'submit':
        ids_train = IDS_LABELED
        ids_test = IDS_SUBMIT
    else:
        assert False

    trainset = SolarMapDatas(ids=ids_train, **kwargs)
    testset = SolarMapDatas(ids=ids_test, **kwargs)
    return trainset, testset


class SolarMapModel():
    """Wrapper data & CNN."""

    def __init__(self, trainset, testset,
                 CNN_type=ShortNet,
                 batch_size=4,
                 num_workers=4,
                 ):

        self.trainset = trainset
        self.testset = testset

        self.mode = self.testset.mode
        self.width = self.testset.width
        self.height = self.testset.height

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.CNN_type = CNN_type

    def train(self, **hyper_param):
        self.cnn = self.CNN_type(input_shape=(3, self.width, self.height))
        num_epochs = hyper_param.get('num_epochs', 5)
        criterion = hyper_param.get('criterion', nn.CrossEntropyLoss())
        optimizer_func = hyper_param.get('optimizer', optim.SGD)
        optimizer = optimizer_func(self.cnn.parameters(),
                                   lr=hyper_param.get('learning_rate', 0.001),
                                   # momentum=hyper_param.get('momentum', 0.9),
                                   )

        logging.info('Beginning Training ...')
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            logging.info('Starting training for epoch={}'.format(epoch))

            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                inputs, labels = data  # get the inputs

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()  # zero the parameter gradients
                output = self.cnn(inputs)  # forward

                # criterion expect a class index from 0 to n_classes-1
                loss = criterion(output, labels - 1)

                # Make some space:
                del output
                del inputs
                del labels

                loss.backward()
                optimizer.step()

                # log statistics
                running_loss += loss.data[0]
                if i % 20 == 19:    # print every 200 mini-batches
                    logging.info('[{epoch}, {i}] loss: {loss}'.format(
                        epoch=epoch, i=i, loss=round(running_loss, 1)))
                    running_loss = 0.0

        logging.info('Finished Training')

    def compute_prediction(self):
        print('Beginning Predicting ...')
        batch_size = self.testloader.batch_size

        df_pred_prob = pd.DataFrame(
            index=pd.Index(self.testset.ids),
            columns=list(CLASSES.values()))
        df_pred = pd.DataFrame(
            0,
            index=pd.Index(self.testset.ids),
            columns=list(CLASSES.values()))

        for i, data in enumerate(self.testloader):
            # get the inputs
            if self.mode == 'train-test':
                inputs, labels = data
            elif self.mode == 'submit':
                inputs = data

            # Compute output:
            output = self.cnn(Variable(inputs))

            # idx:
            idx = i * batch_size

            # Get probabilities:
            pred_proba = F.softmax(output)
            df_pred_prob.iloc[idx: idx + batch_size] = pred_proba.data.numpy()

            # Get highest proba as class:
            _, pred = torch.max(output, 1)
            pred += 1  # Related to -1 in criterion
            for j in range(len(pred)):
                df_pred.iloc[idx + j][CLASSES[pred.data[j]]] = 1

            # if i % 200 == 199:    # print every 2000 mini-batches
            #     print('Predicting {image}th image:  {percentage}% ...'.
            #           format(image=i + 1, percentage=len(self.testloader) / i))

        self.df_pred = df_pred
        self.df_pred_prob = df_pred_prob

    def compute_scores(self):
        """Compute average & area under curve ROC. Only possible if 'train-test'
        mode."""
        assert self.mode == 'train-test'

        df_expected = self.testset.df_classe
        n_pred = self.df_pred.shape[0]
        n_true = 0
        score = 0
        for i in range(n_pred):
            if (self.df_pred.iloc[i] == df_expected.iloc[i]).all():
                n_true += 1

            score += average_precision_score(
                df_expected.iloc[i].tolist(),
                self.df_pred_prob.iloc[i].tolist(),
                average='micro'
            )

        self.accuracy = n_true / n_pred
        self.score = score / n_pred

        print('Accuracy : {acc}%'.format(acc=round(self.accuracy, 2)))
        print('Micro-averave Precision : {score}'
              .format(score=round(self.score, 2)))

    def write_submission_file(self):
        """Write submission file inside submission/ directory with standardized
        name & informations on self.cnn used.
        """
        now = datetime.now()
        df_scores = self.df_pred_prob.copy()
        df_scores.index.name = 'id'
        df_scores.columns = CLASSES.keys()
        path_res = SUBMISSION_DIR / \
            'sub_{now}.csv'.format(now=now.strftime('%d_%m_%Y_H%H_%M_%S'))
        path_cnn = SUBMISSION_DIR / \
            'net_{now}.txt'.format(now=now.strftime('%d_%m_%Y_H%H_%M_%S'))

        f_cnn = open(path_cnn.as_posix(), 'w')
        f_cnn.write(str(self.cnn))
        df_scores.to_csv(path_res)

    def process(self, **hyper_param):
        self.train(**hyper_param)
        self.compute_prediction()
        if self.mode == 'train-test':
            self.compute_scores()


# if __name__ == '__main__':
#     # Parameters for pre-processing:
#     preproc_param = dict(
#         width=96,
#         height=96,
#         transform=transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#              ]),
#         limit_load=None,
#     )
#     trainset, testset = generate_train_test_sets(
#         mode='train-test', **preproc_param)
#     model = SolarMapModel(
#         trainset, testset,
#         batch_size=100,
#         num_workers=4,
#     )

#     # Hyper Parameters for the CNN model
#     hyper_param = dict(
#         num_epochs=5,
#         learning_rate=0.001,
#         criterion=nn.CrossEntropyLoss(),
#         optimizer=optim.Adam,
#     )
#     model.process(**hyper_param)
