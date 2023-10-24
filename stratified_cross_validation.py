# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:58:09 2023

@author: wes_c
"""

import os
import shutil
import tensorflow as tf
import numpy as np

from math import ceil

class StratifiedCrossValidation:
    def __init__(self, dataset, slices):
        #dataset = [[data1,...] [data2,...], ... [data3,...]]
        self.slices = slices
        self.dataset = dataset
        self.folders = list(set([ele.split('/')[-1].split('\\')[0]\
                        for y in dataset for ele in y]))
        self.folders.sort()
        self.size = sum(len(data) for data in dataset)
        self.labels = len(dataset)
        self.lens = [len(data) for data in dataset]
        self.idx = 0
        self.slice_size = [ceil(size / slices) for size in self.lens]

    def _get_slice(self, slice_):
        range_tuples = [(slice_*slice_size, (slice_+1)*slice_size)\
                        for slice_size in self.slice_size]
        train_sets = [data[0:rgs[0]] + data[rgs[1]:] \
                      for rgs, data in zip(range_tuples, self.dataset)]
        test_sets = [data[rgs[0]:rgs[1]]\
                     for rgs, data in zip(range_tuples, self.dataset)]
        return train_sets, test_sets

    def _make_labelled_data(self, data):
        #data = [[data1,...],...]
        labelled_set = []
        labels = np.array(range(self.labels))
        for idx, subdata in enumerate(data):
            for filename in subdata:
                labelled_set.append((filename, (idx == labels).astype(int)))
        return labelled_set

    def _make_folders(self):
        #a rather ugly brute-force solution
        for folder in self.folders:
            if not os.path.exists(f'../data/processed/train/{folder}/'):
                os.makedirs(f'../data/processed/train/{folder}/')
            if not os.path.exists(f'../data/processed/test/{folder}/'):
                os.makedirs(f'../data/processed/test/{folder}/')
        return

    def _wipe_folders(self):
        if os.path.exists('../data/processed/train/'):
            shutil.rmtree('../data/processed/train')
        if os.path.exists('../data/processed/test/'):
            shutil.rmtree('../data/processed/test')
        return

    def _make_filename(self, file):
        parts = file.split('/')
        folder, filename = parts[-1].split('\\')
        return '/'.join([folder, filename])

    def _write_file(self, src_name, dest_name):
        shutil.copy2(src_name, dest_name)
        return

    def _write_files(self, files, train=True):
        #flatten files
        if len(files) < 2:
            raise ValueError("Files have only one label!")
        files = [x for y in files for x in y]
        if train:
            filepath = '../data/processed/train/'
        else:
            filepath = '../data/processed/test/'
        for file in files:
            dest_name = filepath + self._make_filename(file)
            self._write_file(file, dest_name)
        return

    def read_file(self, file, label=False):
        img = tf.io.decode_jpeg(tf.io.read_file(file[0]), channels=3)
        if label:
            return img, file[1] #label
        else:
            return img

    def _make_x_y(self, labelled):
        X_tr = tf.convert_to_tensor([self.read_file(img) for img in labelled])
        y_tr = tf.convert_to_tensor([label[1] for label in labelled])
        return X_tr, y_tr

    def make_tf_sets(self, slice_):
        tr, te = self._get_slice(slice_)
        tr = self._make_labelled_data(tr)
        te = self._make_labelled_data(te)
        X_tr, y_tr = self._make_x_y(tr)
        X_te, y_te = self._make_x_y(te)
        return X_tr, y_tr, X_te, y_te

    def write_slice(self, slice_):
        tr, te = self._get_slice(slice_)
        #first wipe any previous files
        self._wipe_folders()
        self._make_folders()
        self._write_files(tr)
        self._write_files(te, train=False)
        return

    def get_next(self, write=True):
        if self.idx >= self.slices:
            raise ValueError("slice index is greater than slices")
        if not write:
            X_tr, y_tr, X_te, y_te = self.make_tf_sets(self.idx)
            self.idx += 1
            return X_tr, y_tr, X_te, y_te
        else:
            self.write_slice(self.idx)
            self.idx += 1
            return
