# -*- coding: utf-8 -*-
# Copyright (c) 2020-Present Mark Spicer

import os
import torch
import torch.nn as nn
import torch.optim as optim

from image_classifier import net
from image_classifier import data
from image_classifier import constants


class Trainer(object):
    def __init__(self, num_epoch):
        self.num_epoch = num_epoch
        self.data = data.Data()
        self.net = net.Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        print("Started training")

        for epoch in range(self.num_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.data.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
        
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        
        print("Finished training")

    def load(self):
        if os.path.exists(constants.MODEL_PATH):
            self.net.load_state_dict(torch.load(constants.MODEL_PATH))

    def save(self):
        torch.save(self.net.state_dict(), constants.MODEL_PATH)
