# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 11:13:08 2023

@author: wes_c
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import sklearn.metrics as metrics

#alternatively could boilerplate and use a wrapper

class CategoricalPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_precision', **kwargs):
        super(CategoricalPrecision, self).__init__(name=name, **kwargs)
        self.cp = self.add_weight(name='cp', initializer='zeros')
        self.ctr = self.add_weight(name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)
        score = metrics.precision_score(y_true.numpy(),
                                        y_pred.numpy(),
                                        average='weighted',
                                        zero_division=1)
        self.cp.assign_add(score)
        self.ctr.assign_add(1)
        return

    def result(self):
        return self.cp / self.ctr

class CategoricalRecall(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_recall', **kwargs):
        super(CategoricalRecall, self).__init__(name=name, **kwargs)
        self.cr = self.add_weight(name='cr', initializer='zeros')
        self.ctr = self.add_weight(name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)
        score = metrics.recall_score(y_true.numpy(),
                                        y_pred.numpy(),
                                        average='weighted',
                                        zero_division=1)
        self.cr.assign_add(score)
        self.ctr.assign_add(1)
        return

    def result(self):
        return self.cr / self.ctr

class CategoricalF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='categorical_f1', **kwargs):
        super(CategoricalF1Score, self).__init__(name=name, **kwargs)
        self.cf = self.add_weight(name='cf', initializer='zeros')
        self.ctr = self.add_weight(name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)
        score = metrics.f1_score(y_true.numpy(),
                                 y_pred.numpy(),
                                 average='weighted',
                                 zero_division=1)
        self.cf.assign_add(score)
        self.ctr.assign_add(1)

    def result(self):
        return self.cf / self.ctr

class CategoricalAccuracy(tf.keras.metrics.Metric):
    #A DEBUG FUNCTION TO COMPARE WITH KERAS
    def __init__(self, name='categorical_accuracy', **kwargs):
        super(CategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.corr = self.add_weight(name='cp', initializer='zeros')
        self.ctr = self.add_weight(name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)
        score = metrics.accuracy_score(y_true.numpy(),
                                       y_pred.numpy())
        self.corr.assign_add(score)
        self.ctr.assign_add(1)
        return

    def result(self):
        return self.corr / self.ctr
