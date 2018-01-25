#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
from datetime import datetime

from handle_data import (SolarMapVisu,
                         IDS_TRAIN, IDS_SUBMIT, IDS, CLASSES,
                         SUBMISSION_DIR)

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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 4)

    def forward(self, x, training=True):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SolarMapModel():
    def __init__(self, cnn, mode='train-test', **kwargs):

        if mode == 'train-test':
            percentage_train = 70. / 100.
            lst_ids_train = random.sample(
                set(IDS_TRAIN), int(len(IDS_TRAIN) * percentage_train))
            lst_ids_train = pd.Index(lst_ids_train)
            lst_ids_test = IDS_TRAIN.difference(lst_ids_train)

            self.lst_ids_train = lst_ids_train
            self.lst_ids_test = lst_ids_test

        elif mode == 'submit':
            self.lst_ids_train = IDS_TRAIN
            self.lst_ids_test = IDS_SUBMIT

        else:
            assert False

        self.mode = mode

        self.trainset = SolarMapVisu(lst_ids=self.lst_ids_train, **kwargs)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=4)

        self.testset = SolarMapVisu(lst_ids=self.lst_ids_test, **kwargs)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=4)

        self.cnn = cnn

        self.df_classe_train = self.trainset.df_classe
        if mode == 'train-test':
            self.df_classe_test = self.testset.df_classe

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.cnn.parameters(), lr=0.001, momentum=0.9)

        print('Beginning Training ...')
        for epoch in range(10):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 200 == 199:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def compute_prediction(self):
        print('Beginning Predicting ...')
        predicted_dict = {}
        predicted = []
        ids_images = self.lst_ids_test
        num_workers = self.testloader.num_workers
        for i, data in enumerate(self.testloader):
            # get the inputs
            if self.mode == 'train-test':
                inputs, labels = data
            elif self.mode == 'submit':
                inputs = data

            outputs = self.cnn(Variable(inputs))
            _, pred = torch.max(outputs.data, 1)

            predicted += pred.tolist()
            for j in range(len(pred)):
                predicted_dict[ids_images[i * num_workers + j]] = pred[j]

            if i % 200 == 199:    # print every 2000 mini-batches
                print('Predicting {image}th image:  {percentage}% ...'.
                      format(image=i + 1, percentage=len(self.testloader) / i))

        self.predicted = predicted
        self.predicted_dict = predicted_dict

    def compute_scores(self, ):
        """Compute average & area under curve ROC. Only possible if 'train-test'
        mode."""
        assert self.mode == 'train-test'
        self.one_to_four_class()

        y_true = self.testset.labels
        y_scores = self.predicted
        # Accuracy compute:
        tmp = []
        for i in range(len(y_true)):
            tmp.append(y_true[i] == y_scores[i])

        self.accuracy = sum(tmp) / len(y_true)

        score = 0
        for i in range(self.testset.size):
            score += average_precision_score(
                self.df_classe_pred.iloc[i].tolist(),
                self.df_classe_test.iloc[i].tolist(),
                average='micro'
            )

        self.score = score / self.testset.size

        print('Accuracy : {acc}%'.format(acc=round(self.accuracy, 2)))
        print('Micro-averave Precision : {score}'
              .format(score=round(self.score, 2)))

    def one_to_four_class(self):
        """Function in the case of only predicting a class (i.e 0 or 1) to create
        the pandas dataframe equivalent to compare with testset.df_classe.
        """
        df_classe_pred = pd.DataFrame(
            index=pd.Index(self.testset.lst_ids),
            columns=list(CLASSES.values()))

        for key in self.predicted_dict:
            df_classe_pred.loc[key][CLASSES[self.predicted_dict[key]]] = 1.

        self.df_classe_pred = df_classe_pred.fillna(0.)

    def write_submission_file(self):
        """Write submission file inside submission/ directory with standardized
        name & informations on self.cnn used.
        """
        now = datetime.now()
        df_scores = self.df_classe_pred.copy()
        df_scores.index.name = 'id'
        df_scores.columns = CLASSES.keys()
        path_res = SUBMISSION_DIR / \
            'sub_{now}.csv'.format(now=now.strftime('%d_%m_%Y_H%H_%M_%S'))
        path_cnn = SUBMISSION_DIR / \
            'net_{now}.txt'.format(now=now.strftime('%d_%m_%Y_H%H_%M_%S'))

        f_cnn = open(path_cnn.as_posix(), 'w')
        f_cnn.write(str(self.cnn))

        self.df_classe_pred.to_csv(path_res)

        pass


net = CNN()

# Hyper Parameters
hyper_param = dict(
    num_epochs=5,
    batch_size=100,
    learning_rate=0.001,
    criterion=nn.CrossEntropyLoss(),
)

optimizer = torch.optim.Adam(net.parameters(), lr=hyper_param['learning_rate'])


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

qmodel = SolarMapModel(cnn=net, transform=transform, limit_load=100)
qmodel.train()
qmodel.compute_prediction()
# model = SolarMapModel(cnn=net, transform=transform)

# model = SolarMapModel(cnn=net, mode='submit', transform=transform)
