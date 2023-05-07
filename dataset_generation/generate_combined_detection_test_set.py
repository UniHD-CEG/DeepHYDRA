#!/usr/bin/env python3

# 
# Modifications copyright (C) 2023 CERN for the benefit of the ATLAS collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The anomaly injection code ist based in 
# https://github.com/datamllab/tods/blob/benchmark/benchmark/synthetic/Generator/multivariate_generator.py
# the accompanying repository for the paper "TODS: An Automated Time Series Outlier Detection System".

import math
import re
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

anomaly_duration = 20

max_val = 100


def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2*np.pi*freq*timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp*noise
    value = coef*value + offset
    return value


def cosine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.cos(2*np.pi*freq*timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp*noise
    value = coef*value + offset
    return value


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1/(2*i + 1)*sine(length=length, freq=freq*(2*i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value


def collective_global_synthetic(length, base, coef=1.5, noise_amp=0.005):
    value = []
    norm = np.linalg.norm(base)
    base = base/norm
    num = int(length/len(base))
    for i in range(num):
        value.extend(base)
    residual = length - len(value)
    value.extend(base[:residual])
    value = np.array(value)
    noise = np.random.normal(0, 1, length)
    value = coef*value + noise_amp*noise
    return value


def generate_brownian_bridge(size: int) -> np.array:

    wiener = np.cumsum(np.random.normal(size=size - 1))
    wiener = np.pad(wiener, (1, 0))

    tau = np.linspace(0, 1, size)

    return wiener - tau*wiener[-1]


def _parse_channel_name(channel_name):
    parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
    return parameters[1]


def create_channel_names(median_labels, stdev_labels):

    median_labels = ['m_{}'.format(median_label)\
                        for median_label in median_labels]

    stdev_labels = ['std_{}'.format(stdev_label)
                        for stdev_label in stdev_labels]

    labels = np.concatenate((median_labels,
                                stdev_labels))

    return labels


def remove_undetectable_anomalies(data: np.array,
                                        label: np.array):

    rows, cols = data.shape

    for row in range(rows):
        for col in range(cols):
            if (label[row, col] > 0) and\
                    (np.allclose(data[row, :], 0, atol=0.5)):
                label[row, col] = 0
    return label


class MultivariateDataGenerator:
    def __init__(self,
                    dataset: pd.DataFrame,
                    window_size_min: int = 50,
                    window_size_max: int = 200):

        self.window_size_min = window_size_min
        self.window_size_max = window_size_max

        self.window_size_avg = window_size_min +\
                                window_size_max//2 

        self.dim = len(dataset.columns)
        self.dataset_length = len(dataset)

        self.dataset = dataset.to_numpy(copy=True)
        self.dataset_unmodified = self.dataset.copy()

        self.columns = dataset.columns.copy()
        self.timestamps = dataset.index.copy()

        self.labels = np.zeros_like(self.dataset, dtype=np.uint16)

        self.rack_numbers = [_parse_channel_name(column) for column in self.columns]
    
        dataset = None


    def get_dataset_np(self) -> np.array:
        return self.dataset


    def get_dataset_unmodified_numpy(self) -> np.array:
        return self.dataset_unmodified
    

    def get_columns_pd(self) -> pd.Series:
        return self.columns


    def get_timestamps_pd(self) -> pd.Index:
        return self.timestamps


    def get_labels_np(self) -> np.array:
        return self.labels


    def get_columns_in_rack(self, rack_number):
        return [column for column in range(self.dim) if
                                        self.rack_numbers[column] == rack_number]


    def point_global_outliers(self,
                                rack_count,
                                ratio,
                                factor,
                                radius,
                                ignored_racks=None):

        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in self.rack_numbers if rack not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        positions = (np.random.rand(round(self.dataset_length*ratio))*self.dataset_length).astype(int)

        for pos in tqdm(positions):
            racks = list(np.random.choice(rack_numbers_available_racks, rack_count, replace=False))

            for rack in racks:
                columns = self.get_columns_in_rack(rack)
                
                for column in columns:

                    maximum = np.nanmax(self.dataset_unmodified[:, column])
                    minimum = np.nanmin(self.dataset_unmodified[:, column])

                    local_std = np.nanstd(self.dataset_unmodified[max(0, pos - radius):\
                                            min(pos + radius, self.dataset_length), column])

                    self.dataset[pos, column] = self.dataset_unmodified[pos, column]*factor*local_std

                    if 0 <= self.dataset[pos, column] < maximum:
                        self.dataset[pos, column] = maximum
                    if 0 > self.dataset[pos, column] > minimum:
                        self.dataset[pos, column] = minimum
                    self.labels[pos, column] = 0b0000001

        return racks
        

    def point_contextual_outliers(self,
                                    rack_count,
                                    ratio,
                                    factor,
                                    radius,
                                    ignored_racks=None):

        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in self.rack_numbers if rack not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        positions = (np.random.rand(round(self.dataset_length*ratio))*\
                                                    self.dataset_length).astype(int)

        for pos in tqdm(positions):
            racks = list(np.random.choice(rack_numbers_available_racks,
                                                                rack_count,
                                                                replace=False))

            rand_val = min(0.95, abs(np.random.normal(0, 1)))

            for rack in racks:
                columns = self.get_columns_in_rack(rack)

                for column in columns:
                
                    maximum = np.nanmax(self.dataset_unmodified[:, column])
                    minimum = np.nanmin(self.dataset_unmodified[:, column])

                    local_std = np.nanstd(self.dataset_unmodified[max(0, pos - radius):\
                                            min(pos + radius, self.dataset_length), column])
                    
                    self.dataset[pos, column] = self.dataset_unmodified[pos, column]*factor*local_std

                    if self.dataset[pos, column] > maximum: 
                        self.dataset[pos, column] = maximum*rand_val
                    if self.dataset[pos, column] < minimum:
                        self.dataset[pos, column] = minimum*rand_val

                    self.labels[pos, column] = 0b0000010

        return racks


    def persistent_global_outliers(self,
                                    rack_count,
                                    ratio,
                                    factor,
                                    radius,
                                    ignored_racks=None):

        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in self.rack_numbers if rack not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        positions = (np.random.rand(round(self.dataset_length*ratio/\
                                                    self.window_size_avg))*self.dataset_length).astype(int)

        for pos in tqdm(positions):
            racks = list(np.random.choice(rack_numbers_available_racks, rack_count, replace=False))

            radius = np.random.randint(self.window_size_min,
                                            self.window_size_max)//2

            start, end = max(0, pos - radius), min(self.dataset_length, pos + radius)

            brownian_bridge = generate_brownian_bridge(end - start)

            for rack in racks:
                columns = self.get_columns_in_rack(rack)
                
                for column in columns:

                    maximum = np.nanmax(self.dataset_unmodified[:, column])
                    minimum = np.nanmin(self.dataset_unmodified[:, column])

                    local_std = np.nanstd(self.dataset_unmodified[max(0, pos - 2*radius):\
                                            min(pos + 2*radius, self.dataset_length), column])
                    
                    self.dataset[start:end, column] =\
                                    self.dataset_unmodified[pos, column]*\
                                                            brownian_bridge*\
                                                            factor*\
                                                            local_std

                    for index in range(start, end):
                        if 0 <= self.dataset[index, column] < maximum:
                            self.dataset[index, column] = maximum
                        if 0 > self.dataset[index, column] > minimum:
                            self.dataset[index, column] = minimum
                    
                    self.labels[start:end, column] = 0b0000100

        return racks


    def persistent_contextual_outliers(self,
                                        rack_count,
                                        ratio,
                                        factor,
                                        radius,
                                        ignored_racks=None):

        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in self.rack_numbers if rack not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        positions = (np.random.rand(round(self.dataset_length*ratio/\
                                                    self.window_size_avg))*self.dataset_length).astype(int)

        for pos in tqdm(positions):
            racks = list(np.random.choice(rack_numbers_available_racks,
                                                                rack_count,
                                                                replace=False))

            radius = np.random.randint(self.window_size_min,
                                            self.window_size_max)//2

            start, end = max(0, pos - radius), min(self.dataset_length, pos + radius)

            brownian_bridge = generate_brownian_bridge(end - start)

            for rack in racks:
                columns = self.get_columns_in_rack(rack)

                for column in columns:
                
                    maximum = np.nanmax(self.dataset_unmodified[:, column])
                    minimum = np.nanmin(self.dataset_unmodified[:, column])

                    local_std = np.nanstd(self.dataset_unmodified[max(0, pos - 2*radius):\
                                             min(pos + 2*radius, self.dataset_length), column])

                    self.dataset[start:end, column] =\
                            self.dataset_unmodified[start:end, column]*\
                                                            brownian_bridge*\
                                                            factor*\
                                                            local_std

                    for index in range(start, end):
                        if self.dataset[index, column] > maximum:
                            self.dataset[index, column] = maximum*min(0.95, abs(np.random.normal(0, 1)))
                        if self.dataset[index, column] < minimum:
                            self.dataset[index, column] = minimum*min(0.95, abs(np.random.normal(0, 1)))

                    self.labels[start:end, column] = 0b0001000

        return racks


    def collective_global_outliers(self,
                                    rack_count,
                                    ratio,
                                    option='square',
                                    coef=3.,
                                    noise_amp=0.0,
                                    level=5,
                                    freq=0.04,
                                    base=[0.,], # only used when option=='other'
                                    ignored_racks=None): 

        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in self.rack_numbers if rack not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        positions =\
            (np.random.rand(round(self.dataset_length*ratio/\
                                        self.window_size_avg))*self.dataset_length).astype(int)

        for pos in tqdm(positions):
            racks = list(np.random.choice(rack_numbers_available_racks,
                                                                rack_count,
                                                                replace=False))

            radius = np.random.randint(self.window_size_min,
                                            self.window_size_max)//2

            for rack in racks:
                columns = self.get_columns_in_rack(rack)
                
                for column in columns:

                    start, end = max(0, pos - radius), min(self.dataset_length, pos + radius)

                    valid_option = {'square', 'other'}
                    if option not in valid_option:
                        raise ValueError("'option' must be one of {}.".format(valid_option))

                    if option == 'square':

                        offset = self.dataset_unmodified[start, column]

                        sub_data = square_sine(level=level, length=self.dataset_length, freq=freq,
                                                        coef=coef, offset=offset, noise_amp=noise_amp)
                    else:
                        sub_data = collective_global_synthetic(length=self.dataset_length, base=base,
                                                                        coef=coef, noise_amp=noise_amp)
                        
                    self.dataset[start:end, column] = sub_data[start:end]
                    self.labels[start:end, column] = 0b0010000

        return racks


    def collective_trend_outliers(self,
                                    rack_count,
                                    ratio,
                                    factor,
                                    ignored_racks=None):
        
        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in racks if self.rack_numbers not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        positions =\
            (np.random.rand(round(self.dataset_length*ratio/\
                                            self.window_size_avg))*self.dataset_length).astype(int)

        for pos in tqdm(positions):

            racks = list(np.random.choice(rack_numbers_available_racks,
                                                            rack_count,
                                                            replace=False))
            
            radius = np.random.randint(self.window_size_min,
                                            self.window_size_max)//2

            start, end = max(0, pos - radius), min(self.dataset_length, pos + radius)

            slope = np.random.choice([-1, 1])*factor*np.arange(end - start)

            for rack in racks:
                columns = self.get_columns_in_rack(rack)

                for column in columns:        
                    self.dataset[start:end, column] = self.dataset_unmodified[start:end, column] + slope
                    self.labels[start:end, column] = 0b0100000
        
        return racks


    def intra_rack_outliers(self,
                                ratio_temporal: float,
                                ratio_channels: float,
                                average_duration: float = 10.,
                                stdev_duration: float = 1.,
                                ignored_racks=None) -> None:

        if ignored_racks != None:

            rack_numbers_available_racks =\
                    [rack for rack in rack if self.rack_numbers not in ignored_racks]

        else:
            rack_numbers_available_racks = self.rack_numbers

        rng = np.random.default_rng()

        positions =\
            (np.random.rand(round(self.dataset_length*ratio_temporal/\
                                                        average_duration))*self.dataset_length).astype(int)

        for pos in tqdm(positions):

            rack = np.random.choice(rack_numbers_available_racks, 1)[0]


            radius = max(5, int(rng.normal(average_duration,
                                                stdev_duration)))

            start, end = max(0, pos - radius), min(self.dataset_length, pos + radius)

            columns = self.get_columns_in_rack(rack)

            columns_without_nan = np.array(columns)[(~np.any(np.isnan(
                                    self.dataset_unmodified[start:end, columns]), axis=0))]

            if len(columns_without_nan) == 0:
                continue

            column_count = max(1, math.floor(len(columns_without_nan)*ratio_channels))

            anomaly_type = rng.choice([0, 1])
            factor = rng.choice([-0.5, 0.5])

            columns_selected = np.random.choice(columns_without_nan, column_count)

            trial_count = 0

            while np.any(np.isnan(self.dataset_unmodified[start:end, columns_selected].flatten())):
                trial_count += 1

                columns_selected = np.random.choice(columns_without_nan, column_count)

                if trial_count > 100:
                    break

            for column in columns_selected:
                if anomaly_type == 0:
                    self.dataset[start:end, column] = 0
                else:
                    self.dataset[start:end, column] =\
                            self.dataset[start:end, column]*(1 + factor)
                self.labels[start:end, column] = 0b1000000


if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Unreduced HLT Test Set Generator')

    parser.add_argument('--dataset-dir', type=str, default='../datasets/hlt')
    
    args = parser.parse_args()

    # Load datasets

    train_set_x_df = pd.read_csv(f'{args.dataset_dir}/'\
                                    f'hlt_train_set.csv', index_col=0)
    test_set_x_df = pd.read_csv(f'{args.dataset_dir}/'\
                                    f'hlt_test_set.csv', index_col=0)

    nan_amount_train_unlabeled = 100*pd.isna(train_set_x_df.to_numpy().flatten()).sum()/train_set_x_df.size
    nan_amount_test = 100*pd.isna(test_set_x_df.to_numpy().flatten()).sum()/test_set_x_df.size

    print('NaN amounts original datasets:')
    print(f'\tTrain set: {nan_amount_train_unlabeled:.3f} %')
    print(f'\tTest set: {nan_amount_test:.3f} %')

    train_set_x_df.dropna(axis=0,
                            thresh=50,
                            inplace=True)
    
    test_set_x_df.dropna(axis=0,
                            thresh=50,
                            inplace=True)

    nan_amount_train_unlabeled = 100*pd.isna(train_set_x_df.to_numpy().flatten()).sum()/train_set_x_df.size
    nan_amount_test = 100*pd.isna(test_set_x_df.to_numpy().flatten()).sum()/test_set_x_df.size

    print(f'NaN amount:\n\tTrain set: {nan_amount_train_unlabeled:.3f} %')
    print(f'\tTest set: {nan_amount_test:.3f} %')

    def parse_channel_name(channel_name):
        parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
        return parameters[4]

    tpu_numbers_test = [parse_channel_name(label)\
                            for label in list((test_set_x_df).columns.values)]

    tpu_numbers_test_np = np.array(tpu_numbers_test)

    test_set_x_df.drop(test_set_x_df.columns[np.flatnonzero(tpu_numbers_test_np == 16029)[0]],
                                                                            axis=1, inplace=True)

    tpu_numbers_test = [parse_channel_name(label)\
                                for label in list((test_set_x_df).columns.values)]

    tpu_numbers_test_np = np.array(tpu_numbers_test)

    test_set_x_df.drop(test_set_x_df.columns[np.flatnonzero(tpu_numbers_test_np == 55009)[0]],
                                                                            axis=1, inplace=True)

    train_set_size = len(train_set_x_df)

    # We keep the anomaly injection code for the labeled train set
    # in order to ensure that the random values encountered during
    # test set generation are identical to the test sets used thus far

    train_set_labeled_x_df = train_set_x_df.iloc[4*train_set_size//5:, :]

    anomaly_generator_train_labeled = MultivariateDataGenerator(train_set_labeled_x_df,
                                                                    window_size_min=16,
                                                                    window_size_max=256)

    anomaly_generator_train_labeled.point_global_outliers(rack_count=8,
                                                                ratio=0.005,
                                                                factor=1,
                                                                radius=50)

    anomaly_generator_train_labeled.point_contextual_outliers(rack_count=8,
                                                                    ratio=0.005,
                                                                    factor=1,
                                                                    radius=50)

    anomaly_generator_train_labeled.persistent_global_outliers(rack_count=8,
                                                                    ratio=0.025,
                                                                    factor=1,
                                                                    radius=50)
    
    anomaly_generator_train_labeled.persistent_contextual_outliers(rack_count=8,
                                                                        ratio=0.025,
                                                                        factor=1,
                                                                        radius=50)

    anomaly_generator_train_labeled.collective_global_outliers(rack_count=4,
                                                                    ratio=0.025,
                                                                    option='square',
                                                                    coef=5,
                                                                    noise_amp=5,
                                                                    level=10,
                                                                    freq=0.1)

    anomaly_generator_train_labeled.collective_trend_outliers(rack_count=4,
                                                                    ratio=0.025,
                                                                    factor=0.5)
    
    anomaly_generator_test = MultivariateDataGenerator(test_set_x_df,
                                                        window_size_min=16,
                                                        window_size_max=256)

    anomaly_generator_test.point_global_outliers(rack_count=3,
                                                        ratio=0.005,
                                                        factor=0.5,
                                                        radius=50)
    
    anomaly_generator_test.point_contextual_outliers(rack_count=3,
                                                        ratio=0.005,
                                                        factor=0.5,
                                                        radius=50)

    anomaly_generator_test.persistent_global_outliers(rack_count=3,
                                                            ratio=0.025,
                                                            factor=0.5,
                                                            radius=50)
    
    anomaly_generator_test.persistent_contextual_outliers(rack_count=3,
                                                                ratio=0.025,
                                                                factor=0.5,
                                                                radius=50)

    anomaly_generator_test.collective_global_outliers(rack_count=3,
                                                            ratio=0.025,
                                                            option='square',
                                                            coef=5,
                                                            noise_amp=0.5,
                                                            level=10,
                                                            freq=0.1)

    anomaly_generator_test.collective_trend_outliers(rack_count=3,
                                                            ratio=0.025,
                                                            factor=0.5)

    anomaly_generator_test.intra_rack_outliers(ratio_temporal=0.01,
                                                ratio_channels=0.05,
                                                average_duration=10.,
                                                stdev_duration=1.)

    # Save unreduced test set for testing of combined DBSCAN/Transformer-based
    # detection pipeline

    labels = remove_undetectable_anomalies(
                np.nan_to_num(anomaly_generator_test.get_dataset_np(), copy=False),
                                                anomaly_generator_test.get_labels_np())

    test_set_x_df = pd.DataFrame(anomaly_generator_test.get_dataset_np(),
                                    anomaly_generator_test.get_timestamps_pd(),
                                    test_set_x_df.columns)

    test_set_y_df = pd.DataFrame(labels,
                                    anomaly_generator_test.get_timestamps_pd(),
                                    test_set_x_df.columns)


    test_set_x_df.to_hdf(f'{args.dataset_dir}/unreduced_hlt_test_set_x.h5',
                            key='unreduced_hlt_test_set_x',
                            mode='w')

    test_set_y_df.to_hdf(f'{args.dataset_dir}/unreduced_hlt_test_set_y.h5',
                            key='unreduced_hlt_test_set_y',
                            mode='w')
