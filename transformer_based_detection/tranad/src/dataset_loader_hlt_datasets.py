import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from .data_augmentation import HLTDataTimeseriesAugmentor

import warnings
warnings.filterwarnings('ignore')

dataset_path = '../../datasets/hlt/'

class HLTDataset(Dataset):
    def __init__(self,
                    data_source,
                    variant,
                    mode,
                    inverse,
                    scaling_type,
                    scaling_source,
                    applied_augmentations=[],
                    augmented_dataset_size_relative=1,
                    augmented_data_ratio=0):

        self.data_source = data_source.lower()
        self.variant = variant
        self.mode = mode
        self.inverse = inverse

        if scaling_type == 'standard':
            self.scaler = StandardScaler()
        elif scaling_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise NotImplementedError(
                        'Invalid dataset scaler')

        if mode == 'train' or mode == 'unlabeled_train':

            data_train_set_x_pd = pd.read_hdf(dataset_path +\
                                                f'reduced_hlt_{self.data_source}'
                                                f'_train_set_{self.variant}_x.h5')
            
            data_train_set_x_pd.index =\
                        self._remove_timestamp_jumps(
                                pd.DatetimeIndex(data_train_set_x_pd.index))

            if len(applied_augmentations):

                timeseries_augmentor =\
                    HLTDataTimeseriesAugmentor(applied_augmentations)

                data_train_set_x_pd =\
                    timeseries_augmentor.fit_transform(data_train_set_x_pd,
                                                        augmented_dataset_size_relative*\
                                                                    len(data_train_set_x_pd),
                                                        augmented_data_ratio)

            data_train_set_x_np = data_train_set_x_pd.to_numpy()

            self.scaler.fit(data_train_set_x_np)

            data_x_pd = data_train_set_x_pd
            data_x_np = data_train_set_x_np

        if mode == 'labeled_train':

            data_labeled_train_set_x_pd =\
                                pd.read_hdf(dataset_path +\
                                            f'reduced_hlt_{self.data_source}_'
                                            f'labeled_train_set_{self.variant}_x.h5')

            data_labeled_train_set_x_pd.index =\
                        self._remove_timestamp_jumps(
                                pd.DatetimeIndex(data_labeled_train_set_x_pd.index))

            
            labels_pd = pd.read_hdf(dataset_path +\
                                    f'reduced_hlt_{self.data_source}_'
                                    f'labeled_train_set_{self.variant}_y.h5')

            
            labels_pd.index = self._remove_timestamp_jumps(
                                        pd.DatetimeIndex(labels_pd.index))

            if len(applied_augmentations):
                timeseries_augmentor =\
                        HLTDataTimeseriesAugmentor(applied_augmentations)

                target_size = int(len(data_labeled_train_set_x_pd)*\
                                        augmented_dataset_size_relative)

                data_labeled_train_set_x_pd,\
                                    labels_pd =\
                                        timeseries_augmentor.fit_transform_labeled(
                                                                data_labeled_train_set_x_pd,
                                                                labels_pd,
                                                                augmented_dataset_size_relative*\
                                                                    len(data_labeled_train_set_x_pd),
                                                                augmented_data_ratio)

                data_labeled_train_set_x_pd =\
                    data_labeled_train_set_x_pd.iloc[:target_size, :]

                labels_pd =\
                    labels_pd.iloc[:target_size, :]

            data_labeled_train_set_x_np = data_labeled_train_set_x_pd.to_numpy()

            self.labels = np.any(labels_pd.to_numpy()>=1, axis=1).astype(np.bool8).flatten()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train_set_x_pd =\
                        pd.read_hdf(dataset_path +\
                                        f'reduced_hlt_{self.data_source}_'
                                        f'train_set_{self.variant}_x.h5')

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
            data_x_pd = pd.read_hdf(dataset_path +\
                                        f'reduced_hlt_{self.data_source}_'
                                        f'test_set_{self.variant}_x.h5')

            labels_pd = pd.read_hdf(dataset_path +\
                                        f'reduced_hlt_{self.data_source}_'
                                        f'test_set_{self.variant}_y.h5')


            data_x_pd.index =\
                        self._remove_timestamp_jumps(
                                pd.DatetimeIndex(data_x_pd.index))

            self.labels = np.clip(labels_pd.to_numpy(), 0, 1).astype(np.uint8)

            data_x_np = data_x_pd.to_numpy()

            if scaling_source == 'train_set_fit':
                data_train_set_x_pd = pd.read_hdf(dataset_path +\
                                                    f'reduced_hlt_{self.data_source}_'
                                                    f'train_set_{self.variant}_x.h5')

                data_train_set_x_np = data_train_set_x_pd.to_numpy()

                self.scaler.fit(data_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_x_np)
            else:
                raise RuntimeError(
                        'Invalid dataset scaling source')

        elif mode == 'val':

            data_x_pd =  pd.read_hdf(dataset_path +\
                                        f'reduced_hlt_{self.data_source}_'
                                        f'clean_val_set_{self.variant}_x.h5')

            data_x_pd.index =\
                        self._remove_timestamp_jumps(
                                pd.DatetimeIndex(data_x_pd.index))

            data_x_np = data_x_pd.to_numpy()

            if scaling_source == 'train_set_fit':
                data_train_set_x_pd = pd.read_hdf(dataset_path +\
                                                    f'reduced_hlt_{self.data_source}_'
                                                    f'train_set_{self.variant}_x.h5')

                data_train_set_x_np = data_train_set_x_pd.to_numpy()

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


    def _remove_timestamp_jumps(self,
                                index: pd.DatetimeIndex) -> pd.DatetimeIndex:


        # Replace large timestamp jumps resulting from
        # consecutive datapoints coming from different
        # runs with a delta of 5 s, which is the average
        # update frequency of L1 rate data

        delta = index[1:] - index[:-1]

        index = pd.Series(index)

        for i in range(1, len(index)):
            if delta[i - 1] >= pd.Timedelta(10, unit='s'):


                index[i:] = index[i:] - delta[i - 1] +\
                                pd.Timedelta(5, unit='s')

        index = pd.DatetimeIndex(index)

        return index


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def get_channels(self):
        return len(self.data_x[0, :])


    def get_sequence_length(self):
        return self.seq_len


    def get_data(self):
        return self.data_x


    def get_labels(self):
        return self.labels


    def pickle_scaler(self,
                        filename: str) -> None:
        pickle.dump(self.scaler, open(filename, 'wb'))

