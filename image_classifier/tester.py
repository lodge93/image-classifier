# -*- coding: utf-8 -*-
# Copyright (c) 2020-Present Mark Spicer

import os
import torch

from image_classifier import net
from image_classifier import data
from image_classifier import constants

class Tester(object):
    def __init__(self):
        self.data = data.Data()
        self.net = net.Net()

        self.load()

    def load(self):
        if os.path.exists(constants.MODEL_PATH):
            self.net.load_state_dict(torch.load(constants.MODEL_PATH))

    def test(self):
        print("Testing")

        dataiter = iter(self.data.testloader)
        images, labels = dataiter.next()

        correct = 0
        total = 0
        with torch.no_grad():
            for d in self.data.testloader:
                images, labels = d
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
