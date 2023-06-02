#!/usr/bin/env python3

# Modifications copyright (C) 2023 [ANONYMIZED]
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


import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

anomaly_duration = 20

max_val = 100

def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = np.sin(2*np.pi*freq*timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp*noise
    value = coef*value + offset
    return value


def cosine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
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



def parse_channel_name(channel_name):
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

        self.rack_numbers = [parse_channel_name(column) for column in self.columns]
    
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


if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Reduced HLT Dataset Generator')

    parser.add_argument('--dataset-dir', type=str, default='../datasets/hlt')
    parser.add_argument('--variant', type=str, default='2018')
    
    args = parser.parse_args()

    # Load datasets

    train_set_x_df = pd.read_csv(f'{args.dataset_dir}/hlt_train_set_{args.variant}.csv', index_col=0)
    test_set_x_df = pd.read_csv(f'{args.dataset_dir}/hlt_test_set_{args.variant}.csv', index_col=0)
    val_set_x_df = pd.read_csv(f'{args.dataset_dir}/hlt_val_{args.variant}set.csv', index_col=0)
    
    column_names_train = list((train_set_x_df).columns.values)
    column_names_test = list((test_set_x_df).columns.values)
    column_names_val = list((val_set_x_df).columns.values)

    print(f'Channels train: {len(column_names_train)}')
    print(f'Channels test: {len(column_names_test)}')
    print(f'Channels val: {len(column_names_val)}')

    nan_amount_train_unlabeled = np.mean(np.sum(pd.isna(train_set_x_df.to_numpy()), 1)/train_set_x_df.shape[1])
    nan_amount_test = np.mean(np.sum(pd.isna(test_set_x_df.to_numpy()), 1)/test_set_x_df.shape[1])
    nan_amount_val = np.mean(np.sum(pd.isna(val_set_x_df.to_numpy()), 1)/val_set_x_df.shape[1])

    print('Mean sparsity original datasets:')
    print(f'\tTrain set: {100*nan_amount_train_unlabeled:.3f} %')
    print(f'\tTest set: {100*nan_amount_test:.3f} %')
    print(f'\tVal set: {100*nan_amount_val:.3f} %')

    train_set_x_df.dropna(axis=0,
                            thresh=50,
                            inplace=True)
    
    test_set_x_df.dropna(axis=0,
                            thresh=50,
                            inplace=True)

    val_set_x_df.dropna(axis=0,
                            thresh=50,
                            inplace=True)

    nan_amount_train_unlabeled = np.mean(np.sum(pd.isna(train_set_x_df.to_numpy()), 1)/train_set_x_df.shape[1])
    nan_amount_test = np.mean(np.sum(pd.isna(test_set_x_df.to_numpy()), 1)/test_set_x_df.shape[1])
    nan_amount_val = np.mean(np.sum(pd.isna(val_set_x_df.to_numpy()), 1)/val_set_x_df.shape[1])

    print('Mean sparsity preprocessed:')
    print(f'\tTrain set: {100*nan_amount_train_unlabeled:.3f} %')
    print(f'\tTest set: {100*nan_amount_test:.3f} %')
    print(f'\tVal set: {100*nan_amount_val:.3f} %')

    def get_tpu_number(channel_name):
        parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
        return parameters[4]

    tpu_numbers_train = [get_tpu_number(label) for label in column_names_train]
    tpu_numbers_test = [get_tpu_number(label) for label in column_names_test]
    tpu_numbers_val = [get_tpu_number(label) for label in column_names_val]

    tpu_numbers_train_unique = np.array(list(set(tpu_numbers_train)))
    tpu_numbers_test_unique = np.array(list(set(tpu_numbers_test)))
    tpu_numbers_val_unique = np.array(list(set(tpu_numbers_val)))

    rack_numbers_train = np.floor_divide(tpu_numbers_train, 1000)
    rack_numbers_test = np.floor_divide(tpu_numbers_test, 1000)
    rack_numbers_val = np.floor_divide(tpu_numbers_val, 1000)

    racks_train, counts_train =\
        np.unique(rack_numbers_train, return_counts=True)

    print(f'Unique TPUs train set: {len(tpu_numbers_train_unique)}')
    print(f'Unique TPUs test set: {len(tpu_numbers_test_unique)}')
    print(f'Unique TPUs val set: {len(tpu_numbers_val_unique)}')

    # Reduce and save train set

    rack_data_train_unlabeled_all = []

    columns_reduced_train_unlabeled = None
    keys_last = None

    train_set_size = len(train_set_x_df)

    train_set_unlabeled_x_df = train_set_x_df.iloc[:4*train_set_size//5, :]

    train_set_unlabeled_size = len(train_set_unlabeled_x_df)

    print(f'Train set size total: {train_set_size}\t'
                f'unlabeled: {train_set_unlabeled_size}')

    for count, row_x_data in enumerate(tqdm(train_set_unlabeled_x_df.to_numpy(),
                                                desc='Generating unlabeled train set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_train[index]].append(datapoint)

        rack_median_hlt = {}
        rack_hlt_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_hlt[rack] = np.nanmedian(rack_bucket)
            rack_hlt_stdevs[rack] = np.nanstd(rack_bucket)

        rack_median_hlt = dict(sorted(rack_median_hlt.items()))
        rack_hlt_stdevs = dict(sorted(rack_hlt_stdevs.items()))

        if keys_last != None:
            assert rack_median_hlt.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert rack_median_hlt.keys() == rack_hlt_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = rack_median_hlt.keys()

        if type(columns_reduced_train_unlabeled) == type(None):
            columns_reduced_train_unlabeled = create_channel_names(rack_median_hlt.keys(),
                                                                    rack_hlt_stdevs.keys())

        rack_data_np = np.concatenate((np.array(list(rack_median_hlt.values())),
                                            np.array(list(rack_hlt_stdevs.values()))))

        rack_data_train_unlabeled_all.append(rack_data_np)

    rack_data_train_unlabeled_all_np = np.stack(rack_data_train_unlabeled_all)
    rack_data_train_unlabeled_all_np = np.nan_to_num(rack_data_train_unlabeled_all_np, nan=-1)

    nan_amount_train_unlabeled = 100*pd.isna(rack_data_train_unlabeled_all_np.flatten()).sum()/\
                                                            rack_data_train_unlabeled_all_np.size

    print('NaN amount reduced train set: {:.3f} %'.format(nan_amount_train_unlabeled))

    train_set_unlabeled_x_df = pd.DataFrame(rack_data_train_unlabeled_all_np,
                                                        train_set_unlabeled_x_df.index,
                                                        columns_reduced_train_unlabeled)

    train_set_unlabeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_train_set_{args.variant}_x.h5',
                                        key='reduced_hlt_train_set_x',
                                        mode='w')


    # Inject anomalies into labeled train set for semi-supervised training and reduce

    train_set_labeled_x_df = train_set_x_df.iloc[4*train_set_size//5:, :]

    anomaly_generator_train_labeled = MultivariateDataGenerator(train_set_labeled_x_df,
                                                                    window_size_min=16,
                                                                    window_size_max=256)

    anomaly_generator_train_labeled.point_global_outliers(rack_count=3,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=50)
    
    anomaly_generator_train_labeled.point_contextual_outliers(rack_count=3,
                                                                    ratio=0.005,
                                                                    factor=0.5,
                                                                    radius=50)

    anomaly_generator_train_labeled.persistent_global_outliers(rack_count=3,
                                                                    ratio=0.025,
                                                                    factor=1,
                                                                    radius=50)
    
    anomaly_generator_train_labeled.persistent_contextual_outliers(rack_count=3,
                                                                        ratio=0.025,
                                                                        factor=0.5,
                                                                        radius=50)

    anomaly_generator_train_labeled.collective_global_outliers(rack_count=3,
                                                                    ratio=0.025,
                                                                    option='square',
                                                                    coef=5,
                                                                    noise_amp=0.5,
                                                                    level=10,
                                                                    freq=0.1)

    anomaly_generator_train_labeled.collective_trend_outliers(rack_count=3,
                                                                    ratio=0.025,
                                                                    factor=0.5)

    rack_data_train_labeled_all = []
    rack_labels_train_labeled_all = []

    columns_reduced_train_labeled = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(anomaly_generator_train_labeled.get_dataset_np(),
                                    anomaly_generator_train_labeled.get_labels_np()),
                                total=len(anomaly_generator_train_labeled.get_dataset_np()),
                                desc='Generating labeled train set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_train[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_train[index]].append(label)

        rack_median_hlt = {}
        rack_hlt_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_hlt[rack] = np.nanmedian(rack_bucket)
            rack_hlt_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_hlt = dict(sorted(rack_median_hlt.items()))
        rack_hlt_stdevs = dict(sorted(rack_hlt_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_hlt.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (rack_median_hlt.keys() == rack_hlt_stdevs.keys()) and\
                                (rack_median_hlt.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_hlt.keys()

        if type(columns_reduced_train_labeled) == type(None):
            columns_reduced_train_labeled = create_channel_names(rack_median_hlt.keys(),
                                                            rack_hlt_stdevs.keys())
            
            assert np.array_equal(columns_reduced_train_labeled, columns_reduced_train_unlabeled),\
                                            "Labeled train columns don't match unlabeled train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_hlt.values())),
                                            np.array(list(rack_hlt_stdevs.values()))))

        rack_data_train_labeled_all.append(rack_data_np)

        rack_labels_train_labeled_all.append(np.array(list(rack_labels.values())))

    rack_data_train_labeled_all_np = np.stack(rack_data_train_labeled_all)
    rack_data_train_labeled_all_np = np.nan_to_num(rack_data_train_labeled_all_np, nan=-1)

    nan_amount_train_labeled = 100*pd.isna(rack_data_train_labeled_all_np.flatten()).sum()/\
                                                            rack_data_train_labeled_all_np.size

    print('NaN amount reduced labeled train set: {:.3f} %'.format(nan_amount_train_labeled))

    rack_labels_train_labeled_all_np = np.stack(rack_labels_train_labeled_all)

    rack_labels_train_labeled_all_np = np.concatenate([rack_labels_train_labeled_all_np,\
                                                        rack_labels_train_labeled_all_np],
                                                        axis=1)

    rack_labels_train_labeled_all_np =\
                remove_undetectable_anomalies(rack_data_train_labeled_all_np,
                                                rack_labels_train_labeled_all_np)

    train_set_labeled_x_df = pd.DataFrame(rack_data_train_labeled_all_np,
                                            anomaly_generator_train_labeled.get_timestamps_pd(),
                                            columns_reduced_train_labeled)

    train_set_labeled_y_df = pd.DataFrame(rack_labels_train_labeled_all_np,
                                            anomaly_generator_train_labeled.get_timestamps_pd(),
                                            columns_reduced_train_labeled)

    train_set_labeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_labeled_train_set_{args.variant}_x.h5',
                                    key='reduced_hlt_labeled_train_set_x',
                                    mode='w')

    train_set_labeled_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_labeled_train_set_{args.variant}_y.h5',
                                    key='reduced_hlt_labeled_train_set_y',
                                    mode='w')

    # Inject anomalies into test set and reduce

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
                                                            ratio=0.05,
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
                                                            noise_amp=0.05,
                                                            level=10,
                                                            freq=0.1)

    anomaly_generator_test.collective_trend_outliers(rack_count=3,
                                                            ratio=0.025,
                                                            factor=0.5)


    rack_data_test_all = []
    rack_labels_test_all = []

    columns_reduced_test = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(anomaly_generator_test.get_dataset_np(),
                                    anomaly_generator_test.get_labels_np()),
                                total=len(anomaly_generator_test.get_dataset_np()),
                                desc='Generating test set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_test[index]].append(label)

        rack_median_hlt = {}
        rack_hlt_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_hlt[rack] = np.nanmedian(rack_bucket)
            rack_hlt_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_hlt = dict(sorted(rack_median_hlt.items()))
        rack_hlt_stdevs = dict(sorted(rack_hlt_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_hlt.keys() == keys_last,\
                            'Rack bucket keys changed between slices'

            assert (rack_median_hlt.keys() == rack_hlt_stdevs.keys()) and\
                                (rack_median_hlt.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_hlt.keys()

        if type(columns_reduced_test) == type(None):
            columns_reduced_test = create_channel_names(rack_median_hlt.keys(),
                                                            rack_hlt_stdevs.keys())
            
            assert np.array_equal(columns_reduced_test, columns_reduced_train_unlabeled),\
                                                    "Test columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_hlt.values())),
                                            np.array(list(rack_hlt_stdevs.values()))))

        rack_data_test_all.append(rack_data_np)

        rack_labels_test_all.append(np.array(list(rack_labels.values())))

    rack_data_test_all_np = np.stack(rack_data_test_all)
    rack_data_test_all_np = np.nan_to_num(rack_data_test_all_np, nan=-1)

    nan_amount_test = 100*pd.isna(rack_data_test_all_np.flatten()).sum()/\
                                                    rack_data_test_all_np.size

    print('NaN amount reduced test set: {:.3f} %'.format(nan_amount_test))

    rack_labels_test_all_np = np.stack(rack_labels_test_all)

    rack_labels_test_all_np = np.concatenate([rack_labels_test_all_np,\
                                                rack_labels_test_all_np],
                                                axis=1)
    
    rack_labels_test_all_np =\
                remove_undetectable_anomalies(rack_data_test_all_np,
                                                rack_labels_test_all_np)


    test_set_x_df = pd.DataFrame(rack_data_test_all_np,
                                    anomaly_generator_test.get_timestamps_pd(),
                                    columns_reduced_test)

    test_set_y_df = pd.DataFrame(rack_labels_test_all_np,
                                    anomaly_generator_test.get_timestamps_pd(),
                                    columns_reduced_test)

    test_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_test_set_{args.variant}_x.h5',
                            key='reduced_hlt_test_set_x',
                            mode='w')

    test_set_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_test_set_{args.variant}_y.h5',
                            key='reduced_hlt_test_set_x',
                            mode='w')


    # Reduce and save val set without injected anomalies

    rack_data_clean_val_all = []

    columns_reduced_clean_val = None
    keys_last = None

    for count, row_x_data in enumerate(tqdm(val_set_x_df.to_numpy(),
                                                desc='Generating clean val set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_val[index]].append(datapoint)

        rack_median_hlt = {}
        rack_hlt_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_hlt[rack] = np.nanmedian(rack_bucket)
            rack_hlt_stdevs[rack] = np.nanstd(rack_bucket)

        rack_median_hlt = dict(sorted(rack_median_hlt.items()))
        rack_hlt_stdevs = dict(sorted(rack_hlt_stdevs.items()))

        if keys_last != None:
            assert rack_median_hlt.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert rack_median_hlt.keys() == rack_hlt_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = rack_median_hlt.keys()

        if type(columns_reduced_clean_val) == type(None):
            columns_reduced_clean_val = create_channel_names(rack_median_hlt.keys(),
                                                                rack_hlt_stdevs.keys())

        rack_data_np = np.concatenate((np.array(list(rack_median_hlt.values())),
                                            np.array(list(rack_hlt_stdevs.values()))))

        rack_data_clean_val_all.append(rack_data_np)

    rack_data_clean_val_all_np = np.stack(rack_data_clean_val_all)
    rack_data_clean_val_all_np = np.nan_to_num(rack_data_clean_val_all_np, nan=-1)

    nan_amount_clean_val = 100*pd.isna(rack_data_clean_val_all_np.flatten()).sum()/\
                                                    rack_data_clean_val_all_np.size

    print('NaN amount reduced clean val set: {:.3f} %'.format(nan_amount_clean_val))

    clean_val_set_x_df = pd.DataFrame(rack_data_clean_val_all_np,
                                                val_set_x_df.index,
                                                columns_reduced_clean_val)

    clean_val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_clean_val_set_{args.variant}_x.h5',
                                key='reduced_hlt_clean_val_set_x',
                                mode='w')

    # Reduce val set and inject anomalies

    anomaly_generator_val = MultivariateDataGenerator(val_set_x_df,
                                                        window_size_min=16,
                                                        window_size_max=256)

    anomaly_generator_val.point_global_outliers(rack_count=3,
                                                        ratio=0.005,
                                                        factor=0.5,
                                                        radius=50)
    
    anomaly_generator_val.point_contextual_outliers(rack_count=3,
                                                        ratio=0.005,
                                                        factor=0.5,
                                                        radius=50)

    anomaly_generator_val.persistent_global_outliers(rack_count=3,
                                                            ratio=0.025,
                                                            factor=0.5,
                                                            radius=50)
    
    anomaly_generator_val.persistent_contextual_outliers(rack_count=3,
                                                                ratio=0.025,
                                                                factor=0.5,
                                                                radius=50)

    anomaly_generator_val.collective_global_outliers(rack_count=3,
                                                        ratio=0.025,
                                                        option='square',
                                                        coef=5,
                                                        noise_amp=0.5,
                                                        level=10,
                                                        freq=0.1)

    anomaly_generator_val.collective_trend_outliers(rack_count=3,
                                                        ratio=0.025,
                                                        factor=0.5)


    rack_data_val_all = []
    rack_labels_val_all = []

    columns_reduced_val = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(anomaly_generator_val.get_dataset_np(),
                                    anomaly_generator_val.get_labels_np()),
                                total=len(anomaly_generator_val.get_dataset_np()),
                                desc='Generating dirty val set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_val[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_val[index]].append(label)

        rack_median_hlt = {}
        rack_hlt_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_hlt[rack] = np.nanmedian(rack_bucket)
            rack_hlt_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_hlt = dict(sorted(rack_median_hlt.items()))
        rack_hlt_stdevs = dict(sorted(rack_hlt_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_hlt.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (rack_median_hlt.keys() == rack_hlt_stdevs.keys()) and\
                                (rack_median_hlt.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_hlt.keys()

        if type(columns_reduced_val) == type(None):
            columns_reduced_val = create_channel_names(rack_median_hlt.keys(),
                                                        rack_hlt_stdevs.keys())

            assert np.array_equal(columns_reduced_val, columns_reduced_train_unlabeled),\
                                                    "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_hlt.values())),
                                            np.array(list(rack_hlt_stdevs.values()))))

        rack_data_val_all.append(rack_data_np)

        rack_labels_val_all.append(np.array(list(rack_labels.values())))

    rack_data_val_all_np = np.stack(rack_data_val_all)
    rack_data_val_all_np = np.nan_to_num(rack_data_val_all_np, nan=-1)

    nan_amount_test = 100*pd.isna(rack_data_val_all_np.flatten()).sum()/\
                                                    rack_data_val_all_np.size

    print('NaN amount reduced dirty val set: {:.3f} %'.format(nan_amount_test))

    rack_labels_val_all_np = np.stack(rack_labels_val_all)

    rack_labels_val_all_np = np.concatenate([rack_labels_val_all_np,\
                                                rack_labels_val_all_np],
                                                axis=1)

    rack_labels_val_all_np =\
                remove_undetectable_anomalies(rack_data_val_all_np,
                                                rack_labels_val_all_np)

    val_set_x_df = pd.DataFrame(rack_data_val_all_np,
                                    anomaly_generator_val.get_timestamps_pd(),
                                    columns_reduced_val)

    val_set_y_df = pd.DataFrame(rack_labels_val_all_np,
                                    anomaly_generator_val.get_timestamps_pd(),
                                    columns_reduced_val)

    val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_val_set_{args.variant}_x.h5',
                            key='reduced_hlt_val_set_x',
                            mode='w')

    val_set_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_val_set_{args.variant}_y.h5',
                            key='reduced_hlt_val_set_y',
                            mode='w')


