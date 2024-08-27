import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from utils.data_augmentation import EclipseDataTimeseriesAugmentor
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

dataset_path_local = '../../datasets/eclipse/'


class EclipseDataset(Dataset):
    def __init__(self,
                    variant,
                    mode,
                    size, 
                    features,
                    target,
                    inverse,
                    timeenc,
                    freq,
                    scaling_type,
                    scaling_source,
                    applied_augmentations=[],
                    augmented_dataset_size_relative=1,
                    augmented_data_ratio=0):
        
        self.variant = variant
        self.mode = mode

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.inverse = inverse

        if scaling_type == 'standard':
            self.scaler = StandardScaler()
        elif scaling_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise NotImplementedError(
                        'Invalid dataset scaler')

        if mode == 'train' or mode == 'unlabeled_train':

            data_train_set_x_pd =\
                    pd.read_hdf(dataset_path_local +\
                                    'reduced_eclipse_'
                                    'unlabeled_train_set_'
                                    f'{self.variant}.h5',
                                    key='data')

            data_train_set_x_pd.index =\
                pd.DatetimeIndex(data_train_set_x_pd.index)           

            if applied_augmentations:
                if len(applied_augmentations):

                    timeseries_augmentor =\
                        EclipseDataTimeseriesAugmentor(applied_augmentations)

                    data_train_set_x_pd =\
                        timeseries_augmentor.fit_transform(data_train_set_x_pd)

            data_train_set_x_np = data_train_set_x_pd.to_numpy()

            self.scaler.fit(data_train_set_x_np)

            data_x_pd = data_train_set_x_pd
            data_x_np = data_train_set_x_np

        if mode == 'labeled_train':

            data_labeled_train_set_x_pd =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'labeled_train_set_'
                                        f'{self.variant}.h5',
                                        key='data')

            labels_pd = pd.read_hdf(dataset_path_local +\
                                                'reduced_eclipse_'
                                                'labeled_train_set_'
                                                f'{self.variant}.h5',
                                                key='labels')

            data_labeled_train_set_x_pd.index =\
                pd.DatetimeIndex(data_labeled_train_set_x_pd.index)

            labels_pd.index = pd.DatetimeIndex(labels_pd.index)

            if applied_augmentations:
                if len(applied_augmentations):
                    timeseries_augmentor =\
                        EclipseDataTimeseriesAugmentor(applied_augmentations)

                    target_size = int(len(data_labeled_train_set_x_pd)*\
                                            augmented_dataset_size_relative)

                    data_labeled_train_set_x_pd,\
                                        labels_pd =\
                                            timeseries_augmentor.fit_transform_labeled(
                                                                    data_labeled_train_set_x_pd,
                                                                    labels_pd)

                    data_labeled_train_set_x_pd =\
                        data_labeled_train_set_x_pd.iloc[:target_size, :]

                    labels_pd =\
                        labels_pd.iloc[:target_size, :]

            data_labeled_train_set_x_np = data_labeled_train_set_x_pd.to_numpy()

            self.labels = np.any(labels_pd.to_numpy()>=1, axis=1).astype(np.int8).flatten()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train_set_x_pd =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'unlabeled_train_set_'
                                        f'{self.variant}.h5',
                                        key='data')

                data_unlabeled_train_set_x_np =\
                            data_unlabeled_train_set_x_pd.to_numpy()

                self.scaler.fit(data_unlabeled_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_labeled_train_set_x_np)
            else:
                raise RuntimeError(
                        'Invalid dataset scaling source')

            data_x_pd = data_labeled_train_set_x_pd
            data_x_np = data_labeled_train_set_x_np
            
        elif mode == 'test':

            data_x_pd = pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'test_set_'
                                        f'{self.variant}.h5',
                                        key='data')
        
            labels_pd = pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'test_set_'
                                        f'{self.variant}.h5',
                                        key='labels')

            self.labels = np.any(labels_pd.to_numpy()>=1, axis=1).astype(np.int8).flatten()
            
            data_x_pd.index = pd.DatetimeIndex(data_x_pd.index)

            data_x_np = data_x_pd.to_numpy()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train_set_x_pd =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'unlabeled_train_set_'
                                        f'{self.variant}.h5',
                                        key='data')

                data_train_set_x_np =\
                    data_unlabeled_train_set_x_pd.to_numpy()

                self.scaler.fit(data_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_x_np)
            else:
                raise RuntimeError(
                        'Invalid dataset scaling source')

        elif mode == 'val':

            data_x_pd = pd.read_hdf(dataset_path_local +\
                                            'reduced_eclipse_'
                                            'unlabeled_val_set_'
                                            f'{self.variant}.h5',
                                            key='data')
            
            data_x_pd.index = pd.DatetimeIndex(data_x_pd.index)
            data_x_np = data_x_pd.to_numpy()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train_set_x_pd =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'unlabeled_train_set_'
                                        f'{self.variant}.h5',
                                        key='data')

                data_train_set_x_np =\
                    data_unlabeled_train_set_x_pd.to_numpy()

                self.scaler.fit(data_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_x_np)
            else:
                raise RuntimeError(
                        'Invalid scaling source')

        self.data_x = self.scaler.transform(data_x_np)

        self.shape = self.data_x.shape

        if np.any(np.isnan(self.data_x)):
            raise RuntimeError(f'NaN in transformed {mode} data')

        if self.inverse:
            self.data_y = data_x_np
        else:
            self.data_y = self.data_x

        timestamps = pd.DataFrame(data_x_pd.index)

        timestamps.columns = ['date']

        data_stamp = time_features(timestamps,
                                    timeenc=self.timeenc,
                                    freq=self.freq)

        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len],
                                    self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.mode != 'labeled_train':
            return seq_x,\
                    seq_y,\
                    seq_x_mark,\
                    seq_y_mark
        else:
            label = self.labels[s_begin:s_end]

            return seq_x,\
                        seq_y,\
                        seq_x_mark,\
                        seq_y_mark,\
                        label


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def get_channels(self):
        return len(self.data_x[0, :])


    def get_sequence_length(self):
        return self.seq_len


    def get_labels(self):
        return self.labels


    def pickle_scaler(self,
                        filename: str) -> None:
        pickle.dump(self.scaler, open(filename, 'wb'))
