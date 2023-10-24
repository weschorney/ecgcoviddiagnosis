# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:21:02 2023

@author: wes_c
"""

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D, GlobalMaxPool2D,\
 GlobalAveragePooling2D, Dense, MaxPool2D, LeakyReLU, BatchNormalization,\
 Dropout

########################################################
############## IMPLEMENT SPATIAL #######################
########################################################

class SpatialGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid'):
        super(SpatialGate, self).__init__()
        self.conv = Conv2D(filters, (kernel_size,kernel_size),
                           activation=activation, padding='same')

    def call(self, x):
        #data is (batch, width, height, channels)
        avg_ = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_, max_], axis=-1)
        out = self.conv(x)
        return out

########################################################
############## IMPLEMENT CHANNEL #######################
########################################################

class ChannelGate(Layer):
    def __init__(self, channels, reduction_factor, input_shape=None):
        super(ChannelGate, self).__init__()
        self.model = Sequential([
                Dense(channels // reduction_factor, activation='ReLU'),
                Dense(channels, activation='sigmoid')
                ])

    def call(self, x):
        #data is (batch, width, height, channels)
        x_avg = GlobalAveragePooling2D()(x)
        x_max = GlobalMaxPool2D()(x)
        x = tf.concat([x_avg, x_max], axis=1)
        out = self.model(x)
        out = tf.expand_dims(tf.expand_dims(out, 1), 1)
        return out

########################################################
################# IMPLEMENT CBAM #######################
########################################################

class CBAM(Layer):
    def __init__(self, c_channels, c_rf, c_input,
                 s_filters, s_kernel, s_input, spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial
        self.channel_attention = ChannelGate(c_channels, c_rf, input_shape=c_input)
        self.spatial_attention = SpatialGate(s_filters, s_kernel, input_shape=s_input)

    def call(self, x):
        channel_mask = self.channel_attention(x)
        x = channel_mask * x
        if self.spatial:
            spatial_mask = self.spatial_attention(x)
            x = spatial_mask * x
        return x

########################################################
################# ATTENTION BLOCK ######################
########################################################

class AttentionBlock(Layer):
    def __init__(self, channels, kernel_size=(3,3),
                 input_size=None, rate=0.5):
        super(AttentionBlock, self).__init__()
        reduction_factor = 4 if channels >= 4 else channels
        if input_size is not None:
            self.conv = Conv2D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None
            )
        else:
            self.conv = Conv2D(
                channels,
                kernel_size,
                padding='same',
                activation=None
            )
        self.activation = LeakyReLU()
        self.bn = BatchNormalization()
        self.dp = Dropout(rate=rate)
        self.attention = CBAM(
            channels,
            reduction_factor,
            None,
            1,
            3,
            None,
            spatial=True
        )
        self.maxpool = MaxPool2D()

    def call(self, x):
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.dp(output)
        output = self.attention(output)
        output = self.maxpool(output)
        return output

########################################################
#################### ECA BLOCK #########################
########################################################

class ECABlock(Layer):
    def __init__(self, channels, kernel_size=(3,3),
                 input_size=None, rate=0.5):
        super(ECABlock, self).__init__()
        reduction_factor = 4 if channels >= 4 else channels
        if input_size is not None:
            self.conv = Conv2D(
                    channels,
                    kernel_size,
                    input_shape=input_size,
                    padding='same',
                    activation=None
                )
        else:
            self.conv = Conv2D(
                    channels,
                    kernel_size,
                    padding='same',
                    activation=None
                )
        self.activation = LeakyReLU()
        self.bn = BatchNormalization()
        self.dp = Dropout(rate=rate)
        self.attention = CBAM(
                channels,
                reduction_factor,
                None,
                1,
                3,
                None,
                spatial=False
            )
        self.maxpool = MaxPool2D()

    def call(self, x):
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.dp(output)
        output = self.attention(output)
        output = self.maxpool(output)
        return output

###########################################################
############## MULTI-HEADED ATTENTION BLOCK ###############
###########################################################

class MultiHeadAttention(Layer):
    def __init__(self, channels, heads, kernel_size=(3,3),
                 input_size=None, rate=0.5):
        super(MultiHeadAttention, self).__init__()
        reduction_factor = 4 if channels >= 4 else channels
        if input_size is not None:
            self.conv = Conv2D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None
            )
        else:
            self.conv = Conv2D(
                channels,
                kernel_size,
                padding='same',
                activation=None
            )
        self.activation = LeakyReLU()
        self.bn = BatchNormalization()
        self.dp = Dropout(rate=rate)
        self.attention_heads = [
                CBAM(
                        channels,
                        reduction_factor,
                        None,
                        1,
                        3,
                        None
                        )
                for _ in range(heads)
                ]
        self.maxpool = MaxPool2D()

    def call(self, x):
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.dp(output)
        output = tf.concat([self.attention_heads[i](output)\
                            for i in range(len(self.attention_heads))], axis=-1)
        output = self.maxpool(output)
        return output
