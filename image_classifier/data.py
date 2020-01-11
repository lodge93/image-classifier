# -*- coding: utf-8 -*-
# Copyright (c) 2020-Present Mark Spicer

import torch
import torchvision
import torchvision.transforms as transforms

from image_classifier import constants


class Data(object):
    def __init__(self, download=True, root=constants.DATA_DIR):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=self.transform)

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=self.transform)

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2)

        self.classes = (
            'plane',
            'car',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        )
