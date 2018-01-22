#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd

from handle_data import SolarMapVisu, IDS_TRAIN, IDS_SUBMIT, IDS, CLASSES

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
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 8, 5, stride=1, padding=2)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.layer1 = nn.Linear(4 * 4 * 8, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_classes)

    def forward(self, x, training=True):
        # the autoencoder has 3 con layers and 3 deconv layers (transposed conv). All layers but the last have ReLu
        # activation function
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 4 * 8)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, 0.5, training=training)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, 0.5, training=training)
        x = self.layer3(x)
        return x

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = F.softmax(self.forward(x, training=False))
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs
        #   y: the true labels associated with x
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        acc = 100 * torch.sum(torch.eq(indices.float(),
                                       y.float()).float()) / y.size()[0]
        return acc.cpu().data[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = F.softmax(self.forward(x, training=False))


class SolarMapModel():
    def __init__(self, cnn, **kwargs):
        percentage_train = 70. / 100.
        lst_ids_train = random.sample(
            set(IDS_TRAIN), int(len(IDS_TRAIN) * percentage_train))
        lst_ids_train = pd.Index(lst_ids_train)
        lst_ids_test = IDS_TRAIN.difference(lst_ids_train)

        self.lst_ids_train = lst_ids_train
        self.lst_ids_test = lst_ids_test

        self.trainset = SolarMapVisu(lst_ids=lst_ids_train, **kwargs)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=4)

        self.testset = SolarMapVisu(lst_ids=lst_ids_test, **kwargs)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=True, num_workers=4)

        self.cnn = cnn
        self.mode = self.trainset.mode

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.cnn.parameters(), lr=0.001, momentum=0.9)

        print('Beginning Training ...')
        for epoch in range(2):  # loop over the dataset multiple times

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
            inputs, labels = data

            outputs = self.cnn(Variable(inputs))
            _, pred = torch.max(outputs.data, 1)

            predicted += pred.tolist()
            for j in range(len(pred)):
                predicted_dict[ids_images[i*num_workers+j]] = pred[j]

            if i % 200 == 199:    # print every 2000 mini-batches
                print('Predicting {image}th image:  {precentage}% ...'.
                      format(image=i + 1, percentage=len(self.testloader) / i))

        self.predicted = predicted
        self.predicted_dict = predicted_dict

    def compute_scores(self, ):
        assert self.mode == 'train-test'

        y_true = self.testset.labels
        y_scores = self.predicted

        self.score = average_precision_score(y_true, y_scores, average='micro')

        tmp = []
        for i in range(len(y_true)):
            tmp.append(y_true[i] - y_scores[i])

        self.accuracy = self.pred


        # ids_images = self.testset.lst_ids_test
        # classe_predicted[]
        # for i in range(len(ids_images)):
        #     classe_predicted.append(self.predicted_dict[ids_images[i]]])


    def write_submission_file(self, ):
        pass

    def test(self):
        correct = 0
        total = 0
        print('Beginning Testing...')
        for i, data in enumerate(self.testloader, 0):
            # get the inputs
            inputs, labels = data

            outputs = self.cnn(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        from IPython import embed
        embed()  # Enter Ipython
        self.acc = 100 * correct / total
        print('Accuracy of the network on the {ntest_images} test images: {acc}%'
              .format(ntest_images=len(self.testset), acc=self.acc))


net = Net()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

qmodel = SolarMapModel(cnn=net, transform=transform, limit_load=100)
from IPython import embed; embed() # Enter Ipython
model = SolarMapModel(cnn=net, transform=transform)
