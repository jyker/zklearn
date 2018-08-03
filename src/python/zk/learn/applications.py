from keras.utils import multi_gpu_model

import os
import numpy as np
import warnings

from .base import CONFIG
import keras
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras import backend as K

from typing import Dict


class VGG16:        
    def __init__(self, cfg:Dict):
        self.config = cfg
        self.model = self.construct(cfg)

    def construct(self, cfg):
        KC = CONFIG.KEYS
        img_input = Input(shape=cfg[KC.INPUT_SHAPE])

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(cfg[KC.OUT_FEATURES] // 2, activation='relu', name='fc1')(x)
        x = Dense(cfg[KC.OUT_FEATURES] // 2, activation='relu', name='fc2')(x)
        x = Dense(cfg[KC.NB_CLASS], activation='softmax', name='predictions')(x)

        # create model
        model = Model(img_input, x, name='vgg16')

        return model

    def summary(self):
        return self.model.summary()

    def compile(self, optimizer, loss, metrics, gpus=2):
        model = multi_gpu_model(self.model, gpus)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)

        return model


def VGG16_default_config():
    CK = CONFIG.KEYS
    cfg = {}

    # model
    cfg[CK.NB_CLASS] = 9
    cfg[CK.PADDING] = 'same'
    cfg[CK.ACTIVATION] = 'relu'
    cfg[CK.OUT_FEATURES] = 256
    cfg[CK.DROP_OUT] = 0.5
    # cross valid
    cfg[CK.NB_BLOCKS] = 10
    cfg[CK.TRAIN_BLOCKS] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cfg[CK.VALID_BLOCKS] = [0]
    # resample
    cfg[CK.IS_RESAMPLE] = True
    cfg[CK.RESAMPLE_TARGET] = {'4':500, '5':500, '7':500}
    # trian
    cfg[CK.GPUS] = 2
    cfg[CK.LEARNING_RATE] = 0.01
    cfg[CK.DECAY] = 0.0001
    cfg[CK.MOMENTUM] = 0.9
    cfg[CK.LOSS] = keras.losses.categorical_crossentropy
    cfg[CK.OPTIMIZER] = keras.optimizers.SGD(
        lr=cfg[CK.LEARNING_RATE], 
        decay=cfg[CK.DECAY], 
        momentum=cfg[CK.MOMENTUM])
    cfg[CK.METRICS] = [keras.metrics.categorical_accuracy]
    cfg[CK.BATCHSIZE] = 128
    cfg[CK.EPOCH] = 100

    return cfg