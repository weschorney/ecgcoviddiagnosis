# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 18:49:34 2023

@author: wes_c
"""

import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import pickle

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from data_loader import cross_validate, simple_dataloader, prepare
from models import inception, densenet, ecg_covid, attention_net, eca_net,\
                    eca_small, attention_small, attention_covid_net, eca_small2,\
                    ecg_no_attention
from custom_metrics import CategoricalPrecision, CategoricalAccuracy,\
                            CategoricalRecall, CategoricalF1Score
                            
ORIGINAL_F = ["ECG Images of COVID-19 Patients (250)",
              "ECG Images of Myocardial Infarction Patients (77)",
              "ECG Images of Patient that have abnormal heart beats (548)",
              "ECG Images of Patient that have History of MI (203)",
              "Normal Person ECG Images (859)"]

FOLDERS = ["covid",
           "MI_patients",
           "abnormal",
           "MI_history",
           "normal"]

SMALL = ["normal", "covid"]

TRI = ['normal', 'covid', 'other']

SMALL_ALT = ["Normal Person ECG Images (859)",
             "ECG Images of COVID-19 Patients (250)"]

INPUT_SHAPE = (None, 299, 299, 3)
SIMPLE_INPUT = 299

METRICS = [CategoricalAccuracy(),
           CategoricalPrecision(),
           CategoricalRecall(),
           CategoricalF1Score()]

def train(folders, model_name, n_folds=5, classes=5,
          rate=0.5, augment=True, savestr=''):
    if model_name == 'inception':
        model = inception(classes=classes)
    elif model_name == 'densenet':
        model = densenet(classes=classes)
    elif model_name == 'ecg-covid':
        model = ecg_covid(classes=classes)
    elif model_name == 'eca-small':
        model = eca_small(classes=classes, input_shape=SIMPLE_INPUT, rate=rate)
    elif model_name == 'ecg-no-attention':
        model = ecg_no_attention(classes=classes, rate=rate)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    model.build(input_shape=INPUT_SHAPE)
    train_model(folders, model, model_name, n_folds=n_folds,
                savestr=savestr, augment=augment)
    return

def train_model(folders, model, model_name,
                n_folds=5, augment=True, savestr=''):
    if not os.path.exists('./model_weights/'):
        os.makedirs('./model_weights/')
    if not os.path.exists('./model_results/'):
        os.makedirs('./model_results/')
    #training params
    epochs = int(1e5)  # 100000
    #batch_size = 32
    lr = 1e-3
    minimum_lr = 1e-10
    criterion = tf.keras.losses.CategoricalCrossentropy()
    #data
    scv = cross_validate(folders, n_folds)
    #cross validate
    for n in range(n_folds):
        scv.get_next()
        tr_ds, va_ds, te_ds = simple_dataloader()
        tr_ds = prepare(tr_ds, shuffle=True, augment=augment)
        va_ds = prepare(va_ds)
        te_ds = prepare(te_ds)
        model.compile(loss=criterion,
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics=METRICS, run_eagerly=True)
        model_filepath = "./model_weights/" + model_name + f'_fold{n}' + f'a_{augment}' + f'{savestr}'
        model_weights = model_filepath + '_weights.best.hdf5'
        model_results = "./model_results/" + model_name + f'_fold{n}' + f'a_{augment}' + f'{savestr}'
        train_results = model_results + '_train_history.csv'
        test_results = model_results + '_test_results.csv'
        checkpoint = ModelCheckpoint(model_weights,
                                     monitor="val_loss",
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',  # on acc has to go max
                                     save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.5,
                                      min_delta=0.05,
                                      mode='min',  # on acc has to go max
                                      patience=2,
                                      min_lr=minimum_lr,
                                      verbose=1)
        early_stop = EarlyStopping(monitor="val_loss",  # "val_loss"
                                   min_delta=0.05,
                                   mode='min',  # on acc has to go max
                                   patience=10,
                                   verbose=1)    
        # GPU
        history = model.fit(tr_ds,
                            validation_data=va_ds,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[early_stop,
                                       reduce_lr,
                                       checkpoint])
        train_df = pd.DataFrame.from_dict(history.history)
        train_df.to_csv(train_results, index=False)
        #now evaluate, comment out if developing model UNCOMMENT WHEN TESTING OTHERWISE USE VALIDATION
        #stats = model.evaluate(te_ds)
        #metric_names = [ele.name for ele in METRICS]
        #test_dict = dict(zip(['loss'] + metric_names, stats))
        #test_df = pd.DataFrame([test_dict.values()], columns=test_dict.keys())
        #test_df.to_csv(test_results, index=False)
        #and save predicts too
        #y_pred = model.predict(te_ds)
        #tensors = []
        #for batch in te_ds:
        #    tensors.append(batch[1])
        #tensors = tf.concat(tensors, axis=0)
        #tensors = tensors.numpy()
        #with open(f'{model_results}_predicts.pkl', 'wb') as f:
        #    pickle.dump([y_pred, tensors], f)
    return

if __name__ == '__main__':
    train(SMALL, 'eca-small', classes=2,
          augment=False, savestr='cropped50smallbatch_2class', rate=0.5)
