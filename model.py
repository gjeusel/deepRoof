#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime

from handle_data import (SolarMapVisu,
                         IDS_LABELED, IDS_SUBMIT, ALL_IDS, CLASSES,
                         SUBMISSION_DIR,
                         WIDTH, HEIGHT)

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim

from sklearn.metrics import average_precision_score

# from keras.applications import VGG16
# from keras.optimizers import SGD

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Let's get inspired from http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


def ispair(i):
    return i % 2 == 0


def get_conv_output(shape, layer):
    bs = 1
    x = Variable(torch.rand(bs, *shape))
    out = layer(x)
    return out.size()[1:]


class CNN(nn.Module):
    def __init__(self, input_shape=(3, WIDTH, HEIGHT)):
        super(CNN, self).__init__()

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

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))

        shape = get_conv_output(input_shape, self.layer1)
        shape = get_conv_output(shape, self.layer2)
        shape = get_conv_output(shape, self.layer3)
        # shape = get_conv_output(shape, self.layer4)

        self.fc1 = nn.Linear(shape[0]*shape[1]*shape[2], 120)  # fully connected
        self.fc2 = nn.Linear(120, 84)  # fully connected
        self.fc3 = nn.Linear(84, 4)  # fully connected

    def forward(self, x, training=True):
        out = self.layer1(x)
        out = nn.functional.dropout2d(out, 0.15)
        out = self.layer2(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = self.layer3(out)
        out = nn.functional.dropout2d(out, 0.20)
        # out = self.layer4(out)
        # out = nn.functional.dropout2d(out, 0.20)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class SolarMapModel():
    def __init__(self, cnn, mode='train-test', **kwargs):

        if mode == 'train-test':
            percentage_train = 70. / 100.
            ids_train = random.sample(
                set(IDS_LABELED), int(len(IDS_LABELED) * percentage_train))
            ids_train = pd.Index(ids_train)
            ids_test = IDS_LABELED.difference(ids_train)

            self.ids_train = ids_train
            self.ids_test = ids_test

        elif mode == 'submit':
            self.ids_train = IDS_LABELED
            self.ids_test = IDS_SUBMIT

        else:
            assert False

        self.mode = mode

        self.trainset = SolarMapVisu(ids=self.ids_train, **kwargs)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=4)

        self.testset = SolarMapVisu(ids=self.ids_test, **kwargs)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=4)

        self.cnn = cnn

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.cnn.parameters(), lr=0.001, momentum=0.9)

        logging.info('Beginning Training ...')
        for epoch in range(120):  # loop over the dataset multiple times
            logging.info('Starting training for epoch={}'.format(epoch))

            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self.cnn(inputs)
                # criterion expect a class index from 0 to n_classes-1
                loss = criterion(output, labels - 1)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 20 == 19:    # print every 200 mini-batches
                    logging.info('[{epoch}, {i}] loss: {loss}'.format(
                        epoch=epoch, i=i, loss=round(running_loss, 1)))
                    running_loss = 0.0

        logging.info('Finished Training')

    def compute_prediction(self):
        print('Beginning Predicting ...')
        ids_images = self.ids_test
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

    def compute_scores(self, ):
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

    def process(self):
        self.train()
        self.compute_prediction()
        if self.mode == 'train-test':
            self.compute_scores()


# net = CNN()

# # Hyper Parameters
# hyper_param = dict(
#     num_epochs=5,
#     batch_size=100,
#     learning_rate=0.001,
#     criterion=nn.CrossEntropyLoss(),
# )

# optimizer = torch.optim.Adam(net.parameters(), lr=hyper_param['learning_rate'])

# qmodel = SolarMapModel(cnn=net, transform=transform, limit_load=100)
# qmodel.train()
# qmodel.compute_prediction()
# # model = SolarMapModel(cnn=net, transform=transform)
