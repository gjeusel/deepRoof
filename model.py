#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from handle_data import (DATA_DIR, load_raw_images, get_images_ids,
                         ROOF_NORTH_SOUTH, ROOF_WEST_EAST,
                         ROOF_UNKNOWN, ROOF_FLAT,
                         )

import torch
import torchvision
import torchvision.transforms as transforms

import torch.utils.data as data

import torch.nn as nn
import torch.nn.functional as F

from keras.applications import VGG16
from keras.optimizers import SGD

import PIL
from PIL import Image


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
