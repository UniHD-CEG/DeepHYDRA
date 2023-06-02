import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.omni_anomaly_utils import *
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class OmniAnomalyDataset(Dataset):
    def __init__(self,
                    dataset,
                    mode,
                    size, 
                    features,
                    target,
                    inverse,
                    timeenc,
                    freq,
                    scaling_type,
                    scaling_source):

        self.dataset = dataset
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
            (data, labels), _, _ = get_data(dataset)

            self.scaler.fit(data)

            timestamps = pd.DataFrame(pd.date_range(start='1/1/2018',
                                            periods=len(data),
                                            freq='10s'), columns=['date'])

        elif mode == 'labeled_train':
            (train_set_x, _), _, (data, labels) = get_data(dataset)

            if scaling_source == 'train_set_fit':
                self.scaler.fit(train_set_x)
            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data)
            else:
                raise RuntimeError(
                        'Invalid scaling source')

            timestamps = pd.DataFrame(pd.date_range(start='1/7/2020',
                                            periods=len(data),
                                            freq='10s'), columns=['date'])

        elif mode == 'test':
            (train_set_x, _), (data, labels), _ = get_data(dataset)

            if scaling_source == 'train_set_fit':
                self.scaler.fit(train_set_x)
            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data)
            else:
                raise RuntimeError(
                        'Invalid scaling source')
    
            data_len = len(data)

            data = data[int(data_len*0.9):, :]
            labels = labels[int(data_len*0.9):]

            timestamps = pd.DataFrame(pd.date_range(start='5/7/2020',
                                            periods=len(data),
                                            freq='10s'), columns=['date'])

        elif mode == 'val':
            (data, labels), _, _ = get_data(dataset)

            data_len = len(data)

            train_set_x = data[:int(data_len*0.9), :]

            data = data[int(data_len*0.9):, :]

            if scaling_source == 'train_set_fit':
                self.scaler.fit(train_set_x)
            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data)
            else:
                raise RuntimeError(
                        'Invalid scaling source')

            timestamps = pd.DataFrame(pd.date_range(start='5/7/2020',
                                            periods=len(data),
                                            freq='10s'), columns=['date'])

        self.data_x = self.scaler.transform(data)

        if np.any(np.isnan(self.data_x)):
            raise RuntimeError(f'NaN in transformed {mode} data')

        data_stamp = time_features(timestamps,
                                    timeenc=self.timeenc,
                                    freq=self.freq)

        if self.inverse:
            self.data_y = data
        else:
            self.data_y = self.data_x

        self.data_stamp = data_stamp

        self.labels = labels


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
        pass