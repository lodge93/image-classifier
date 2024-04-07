#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2020-Present Mark Spicer

from setuptools import setup
from setuptools import find_packages


setup(
    name='image-classifier',
    version='0.0.1',
    description='My first image classifier, following the pytorch tutorial',
    url='https://github.com/lodge93/image-classifier',
    author='Mark Spicer',
    author_email='spicer93@gmail.com',
    maintainer='Mark Spicer',
    maintainer_email='spicer93@gmail.com',
    license='MIT',
    packages=find_packages(),
    scripts=[
        'scripts/trainer',
        'scripts/predictor',
        'scripts/tester',
    ],
    install_requires=[
        'matplotlib==3.1.2',
        'numpy==1.22.0',
        'Pillow==6.2.2',
        'torch==1.3.1',
        'torchvision==0.4.2',
    ]
)
