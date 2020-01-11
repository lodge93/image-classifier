# -*- coding: utf-8 -*-
# Copyright (c) 2020-Present Mark Spicer

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from image_classifier import net
from image_classifier import data
from image_classifier import constants


class Predictor(object):
    def __init__(self):
        self.data = data.Data()
        self.net = net.Net()
        
        self.load()

    def load(self):
        if os.path.exists(constants.MODEL_PATH):
            self.net.load_state_dict(torch.load(constants.MODEL_PATH))

    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def predict(self):
        dataiter = iter(self.data.trainloader)
        images, labels = dataiter.next()

        outputs = self.net(images)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % self.data.classes[predicted[j]]
                              for j in range(4)))
        
        # show images
        self.imshow(torchvision.utils.make_grid(images))
