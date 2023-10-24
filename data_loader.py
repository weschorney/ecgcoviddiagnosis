# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:06:46 2023

@author: wes_c
"""

import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from glob import glob
from functools import partial
from stratified_cross_validation import StratifiedCrossValidation

FOLDERS = ["covid",
           "MI_patients",
           "abnormal",
           "MI_history",
           "normal"]

IMAGE_SIZE = (299, 299)

def load_images_from_folder(folder, prefix="../data/processed/"):
    #load img paths and return as list
    file_names = glob(f"{prefix}{folder}/*.jpg")
    return list(file_names)

def load_images(folders, prefix="../data/processed/"):
    dataset = []
    for folder_name in folders:
        file_names = load_images_from_folder(folder_name, prefix=prefix)
        dataset.append(file_names)
    return dataset

def cross_validate(folders, n, prefix="../data/processed/"):
    dataset = load_images(folders, prefix=prefix)
    scv = StratifiedCrossValidation(dataset, n)
    return scv

def get_label_from_path(path):
    print(path)
    if isinstance(path, bytes):
        img_name = path.split(b'\\')[-1]
    else:
        img_name = tf.strings.split(path, b'\\')[-1]
    label = re.search(b'.*(?=\_img[0-9]+)', img_name)
    return label.group(0)

def get_all_classes(ds):
    class_names = []
    for file in ds:
        class_names.append(get_label_from_path(file.numpy()))
    class_names = list(set(class_names))
    class_names.sort()
    return np.array(class_names)

def get_onehot(file_path, class_names):
    label = get_label_from_path(file_path)
    one_hot = label == class_names
    return tf.argmax(one_hot)

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return img

def process_path(class_names, file_path):
    label = get_label_from_path(file_path)
    onehot = get_onehot(label, class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, onehot

def make_tf_dataloaders(prefix='../data/processed/', val_size=0.2):
    list_test = tf.data.Dataset.list_files(prefix+'test/*', shuffle=False)
    list_ds = tf.data.Dataset.list_files(prefix+'train/*', shuffle=False)
    class_names = get_all_classes(list_ds)
    image_count = list_ds.cardinality().numpy()
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    val_size = int(image_count * val_size)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    my_func = partial(process_path, class_names)
    train_ds = train_ds.map(my_func)
    val_ds = val_ds.map(my_func)
    test_ds = list_test.map(my_func)
    return train_ds, val_ds, test_ds

def simple_dataloader(prefix='../data/processed/', val_size=0.2, seed=777,
                      batch_size=4):
    train_ds = tf.keras.utils.image_dataset_from_directory(
            prefix+'train/',
            validation_split=val_size,
            subset='training',
            seed=seed,
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            label_mode='categorical',
            )
    val_ds = tf.keras.utils.image_dataset_from_directory(
            prefix+'train/',
            validation_split=val_size,
            subset='validation',
            seed=seed,
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            label_mode='categorical',
            )
    test_ds = tf.keras.utils.image_dataset_from_directory(
            prefix+'test/',
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            label_mode='categorical',
            )
    return train_ds, val_ds, test_ds

def prepare(ds, shuffle=False, augment=False):
    rescale = tf.keras.Sequential([layers.Rescaling(1./255)])
    data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.05, 0.05, fill_mode='wrap')
            ])
    ds = ds.map(lambda x,y: (rescale(x), y))
    if shuffle:
        ds = ds.shuffle(1000)
    if augment:
        ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y))
    return ds

if __name__ == "__main__":
    scv = cross_validate(FOLDERS, 5)
    scv.get_next(write=True)
    tr_ds, va_ds, te_ds = simple_dataloader()
