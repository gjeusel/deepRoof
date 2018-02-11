#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import pandas as pd
from datetime import datetime
from collections import OrderedDict

from common import (IDS_LABELED, IDS_SUBMIT,
                    CLASSES,
                    SUBMISSION_DIR, TRAINED_DIR,
                    )

from handle_data import SolarMapDatas
from networks import LargeNet, ShortNet
from historic_models import HistoricModel

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import average_precision_score

# from keras.applications import VGG16
# from keras.optimizers import SGD


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

        self.model_db = HistoricModel()

    def train(self, CNN_type, **hyper_param):
        """Train the Neural Network with hyper_param.
        If an already existing Neural Network equivalent, just continue the training
        where it stopped.
        """

        self.cnn = CNN_type(input_shape=(3, self.width, self.height))

        is_model_trained = not self.model_db.inspect_models(
            str(self.cnn), hyper_param, self.width, self.height) is None

        id_model = self.model_db.get_id_model(
            str(self.cnn), hyper_param, self.width, self.height)

        if is_model_trained:
            net_state, optimizer, from_epoch = self.model_db.get_existing_cnn(
                id_model)
            self.cnn.load_state_dict(net_state)
        else:
            from_epoch = 0
            optimizer_func = hyper_param.get('optimizer', optim.SGD)
            optimizer = optimizer_func(self.cnn.parameters(),
                                       lr=hyper_param.get(
                                           'learning_rate', 0.001),
                                       )

        num_epochs = hyper_param.get('num_epochs', 5)
        criterion = hyper_param.get('criterion', nn.CrossEntropyLoss())

        logging.info('Beginning Training ...')
        # loop over the dataset multiple times
        for epoch in range(from_epoch, num_epochs):
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
                if i % 20 == 19:    # print every 20 mini-batches
                    logging.info('[{epoch}, {i}] loss: {loss}'.format(
                        epoch=epoch, i=i + 1, loss=round(running_loss, 1)))
                    running_loss = 0.0

            if epoch % 2 == 0:  # save every 2 epoch
                train_progress = 100 - (num_epochs - (epoch+1))/num_epochs*100
                self.model_db.save_model(
                    id_model,
                    self.cnn, optimizer, epoch+1,
                    hyper_param, self.width, self.height,
                    int(train_progress),
                )

        logging.info('Finished Training')
        return id_model

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

    def process(self, CNN_type, **hyper_param):
        id_model = self.train(CNN_type, **hyper_param)
        self.compute_prediction()
        if self.mode == 'train-test':
            self.compute_scores()
            self.model_db.add_score(id_model, self.accuracy, self.score)
