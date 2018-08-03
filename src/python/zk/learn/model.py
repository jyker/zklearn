import numpy as np
import keras
import os
import tensorflow as tf
from .base import CONFIG
from typing import Dict, Tuple
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Permute, Reshape
from keras.layers import Flatten, Dense, Dropout, Input, Activation, GlobalMaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential
from keras.utils import multi_gpu_model

# keras.backend.set_image_data_format("channels_first")

class ImageBasedMalNet:
    """
    """
    def __init__(self, cfg:Dict):
        self.config = cfg
        self.model = self.construct(cfg)

    def construct(self, config):
        KC = CONFIG.KEYS

        model = Sequential()
        # conv_block_1
        model.add(Conv2D(
            filters=32,
            kernel_size=config[KC.KERNEL_SIZE],
            strides=config[KC.STRIDES],
            padding=config[KC.PADDING],
            input_shape=config[KC.INPUT_SHAPE]))
        model.add(BatchNormalization())
        model.add(Activation(config[KC.ACTIVATION]))
        model.add(MaxPooling2D(
            pool_size=config[KC.KERNEL_SIZE],
            strides=config[KC.POOL_STRIDE],
            padding=config[KC.PADDING]))
        # conv_block_2
        _stride = config[KC.STRIDES]
        model.add(Conv2D(
            filters=64,
            kernel_size=config[KC.KERNEL_SIZE],
            strides=_stride,
            padding=config[KC.PADDING]))
        model.add(BatchNormalization())
        model.add(Activation(config[KC.ACTIVATION]))
        model.add(MaxPooling2D(
            pool_size=config[KC.KERNEL_SIZE],
            strides=config[KC.POOL_STRIDE],
            padding=config[KC.PADDING]))
        # conv_block_3
        model.add(Conv2D(
            filters=96,
            kernel_size=config[KC.KERNEL_SIZE],
            strides=_stride,
            padding=config[KC.PADDING]))
        model.add(BatchNormalization())
        model.add(Activation(config[KC.ACTIVATION]))
        model.add(MaxPooling2D(
            pool_size=config[KC.KERNEL_SIZE],
            strides=config[KC.POOL_STRIDE],
            padding=config[KC.PADDING]))
        # conv_block_4
        model.add(Conv2D(
            filters=128,
            kernel_size=config[KC.KERNEL_SIZE],
            strides=_stride,
            padding=config[KC.PADDING]))
        model.add(BatchNormalization())
        model.add(Activation(config[KC.ACTIVATION]))
        model.add(MaxPooling2D(
            pool_size=config[KC.KERNEL_SIZE],
            strides=config[KC.POOL_STRIDE],
            padding=config[KC.PADDING]))

        _shape = model.outputs[0].shape.as_list()
        model.add(TimeDistributed(Reshape((_shape[-1] * _shape[-2], ))))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(config[KC.DROP_OUT]))

        model.add(Dense(config[KC.OUT_FEATURES]))
        model.add(BatchNormalization())
        model.add(Activation(config[KC.ACTIVATION]))

        model.add(Dense(config[KC.OUT_FEATURES] // 2))
        model.add(BatchNormalization())
        model.add(Activation(config[KC.ACTIVATION]))

        model.add(Dense(config[KC.NB_CLASS], activation='softmax'))

        model = Model(inputs=model.inputs, outputs=model.outputs)

        return model

    def summary(self):
        return self.model.summary()

    def compile(self, optimizer, loss, metrics, gpus=1):
        if gpus == 1:
            model = self.model
        elif gpus >= 1:
            model = multi_gpu_model(self.model, gpus)
        else:
            raise ValueError("valid gpus {}".format(gpus))

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)

        return model


def ImageBasedMalNet_default_config():
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