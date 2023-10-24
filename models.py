# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:39:55 2023

@author: wes_c
"""

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as ls

from attention import AttentionBlock, ECABlock

class PretrainedTop(tf.keras.Model):
    def __init__(self, inception, n_classes):
        super(PretrainedTop, self).__init__()
        self.inception = inception
        #for layer in self.inception.layers:
        #    layer.trainable = False
        self.l1 = ls.Flatten()
        self.l11 = ls.Dense(256, activation='ReLU')
        self.l2 = ls.Dense(n_classes, activation='softmax')

    def call(self, x):
        out = self.inception(x)
        out = self.l2(self.l11(self.l1(out)))
        return out

def inception(classes=5):
    incep = ks.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,)
    model = PretrainedTop(incep, classes)
    return model

def densenet(classes=5):
    dn = ks.applications.densenet.DenseNet201(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,)
    model = PretrainedTop(dn, classes)
    return model

def ecg_covid(classes=5):
    #implementation of ECG-COVID: an end-to-end deep model based on
    #electrocardiogram for COVID-19 detection; here we use more than
    #two classes depending on the application.
    model = ks.Sequential([
            ls.Conv2D(16, (3,3), padding='same'),
            ls.Conv2D(16, (3,3), padding='same', activation='LeakyReLU'),
            ls.MaxPool2D(),
            ls.Conv2D(32, (3,3), padding='same', activation='LeakyReLU'),
            ls.MaxPool2D(),
            ls.Conv2D(64, (3,3), padding='same', activation='LeakyReLU'),
            ls.MaxPool2D(),
            ls.Conv2D(128, (3,3), padding='same', activation='LeakyReLU'),
            ls.MaxPool2D(),
            ls.Conv2D(256, (3,3), padding='same', activation='LeakyReLU'),
            ls.Flatten(),
            ls.Dense(256, activation='ReLU'),
            ls.Dense(classes, activation='softmax')
            ])
    return model

def ecg_no_attention(classes=5, rate=0.5):
    model = ks.Sequential([
            ls.Conv2D(16, (3, 3), padding='same', activation='LeakyReLU'),
            ls.BatchNormalization(),
            ls.Dropout(rate=0.5),
            ls.MaxPool2D(),
            ls.Conv2D(32, (3, 3), padding='same', activation='LeakyReLU'),
            ls.BatchNormalization(),
            ls.Dropout(rate=0.5),
            ls.MaxPool2D(),
            ls.Flatten(),
            ls.Dense(256, activation='LeakyReLU'),
            ls.Dense(classes, activation='softmax')
        ])
    return model

class ECASmall(tf.keras.Model):
    def __init__(self, input_shape=299, classes=5, rate=0.5):
        super(ECASmall, self).__init__()
        self.model = ks.Sequential([
            ECABlock(16, rate=rate),
            ECABlock(32, rate=rate),
            ls.Flatten(),
            ls.Dense(256, activation='LeakyReLU'),
            ls.Dense(classes, activation='softmax')
            ])

    def call(self, x):
        return self.model(x)

#for gflop profile
def ecasmall_func(classes=2, rate=0.5):
    model = ks.Sequential([
            ECABlock(16, rate=rate),
            ECABlock(32, rate=rate),
            ls.Flatten(),
            ls.Dense(256, activation='LeakyReLU'),
            ls.Dense(classes, activation='softmax')
            ])
    return model

def irmak_model(input_shape, classes=2):
    model = ks.Sequential([
            ks.layers.Conv2D(96, (3, 3), input_shape=input_shape,
                             activation='ReLU'),
            ls.BatchNormalization(),
            ls.MaxPool2D(),
            ks.layers.Conv2D(192, (3, 3), activation='ReLU'),
            ls.MaxPool2D(),
            ks.layers.Conv2D(256, (3, 3), activation='ReLU'),
            ls.MaxPool2D(),
            ks.layers.Conv2D(256, (3, 3), activation='ReLU'),
            ls.MaxPool2D(),
            ls.Flatten(),
            ls.Dense(4096, activation='ReLU'),
            ls.Dense(classes, activation='softmax')
            ])
    return model

def eca_small(classes=5, input_shape=299, rate=0.5):
    return ECASmall(input_shape=input_shape, classes=classes, rate=rate)
