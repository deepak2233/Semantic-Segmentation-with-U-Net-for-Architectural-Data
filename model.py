import argparse
import os
import json
import io
import numpy as np
from tensorflow.keras import regularizers
from natsort import natsorted
import tensorflow as tf
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def unet_model1(input_shape, num_classes, dropout_rate):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def unet_model2(input_shape, num_classes, dropout_rate, l1_regularization, l2_regularization):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same",
                   kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = Conv2D(num_classes, 1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


