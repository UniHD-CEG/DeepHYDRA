import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .data_augmentation import EclipseDataTimeseriesAugmentor

import warnings
warnings.filterwarnings('ignore')

dataset_path_local = '../../datasets/eclipse/'


class EclipseDataset(Dataset):
    def __init__(self,
                    variant,
                    mode,
                    inverse,
                    scaling_type,
                    scaling_source,
                    applied_augmentations=[],
                    augmented_dataset_size_relative=1,
                    augmented_data_ratio=0):

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

            labels_pd = self.expand_labels(data_labeled_train_set_x_pd.columns, labels_pd)

            self.labels = (labels_pd.to_numpy()>=1).astype(np.int8)

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

            labels_pd = self.expand_labels(data_x_pd.columns, labels_pd)
            self.labels = (labels_pd.to_numpy()>=1).astype(np.int8)
            
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


    def expand_labels(self,
                        data_columns: pd.Index,
                        labels: pd.DataFrame):
    
        app_names = ['exa', 'lammps', 'sw4', 'sw4lite']

        def _renaming_func(element):

            column_name = ''

            for app_name in app_names:
                if f'{app_name}_' in element:
                    column_name = app_name

            return column_name
        
        label_columns_expanded =\
            pd.Index(data_columns.map(_renaming_func))

        labels_expanded = pd.DataFrame(columns=label_columns_expanded)

        for app_name in app_names:
            labels_expanded[app_name] =\
                            labels.loc[:, app_name]

        labels_expanded.columns = data_columns

        return labels_expanded


    def get_data(self):
        return self.data_x


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def get_channels(self):
        return len(self.data_x[0, :])


    def get_labels(self):
        return self.labels


    def pickle_scaler(self,
                        filename: str) -> None:
        pickle.dump(self.scaler, open(filename, 'wb'))
