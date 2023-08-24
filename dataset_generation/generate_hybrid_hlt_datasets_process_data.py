#!/usr/bin/env python3
import math
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

max_val = 100

image_width = 1920
image_height = 1080

plot_window_size = 100

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255,255,255)
thickness = 1
line_type = 2


def find_timestamp_jumps(index: pd.DatetimeIndex) -> None:

        delta = index[1:] - index[:-1]

        index = pd.Series(index)

        for i in range(0, len(index) - 1):
            if delta[i] >= pd.Timedelta(10, unit='s'):
                # print(f'Found timestamp jump at {i} between '
                #         f'timestamps {index[i]} and {index[i+1]}')
                print(index[i])
                print(index[i+1])


def generate_anomaly_labels(failure_data: pd.DataFrame,
                                        index: pd.Index,
                                        columns: pd.Index,
                                        process_labels: np.array,
                                        prepad: int = 0) -> pd.DataFrame:

    index = pd.DatetimeIndex(index)
    labels = pd.DataFrame(0, index, columns, dtype=np.uint32)

    failure_data = failure_data.droplevel(0)

    for failure in failure_data.itertuples():

        start = pd.DatetimeIndex([failure.start], tz=index.tz)
        end = pd.DatetimeIndex([failure.end], tz=index.tz)
        
        index_following_start =\
            labels.index.get_indexer(start,
                                        method='bfill',
                                        tolerance=pd.Timedelta(5, unit='m'))[0]

        if index_following_start != -1:

            index_following_start =\
                max(0, index_following_start - prepad)

            print('Found start timestamp within tolerance')
            print(f'Anomaly start: {start}')
            print(f'Timestamp within tolerance: '\
                        f'{index[index_following_start]}'\
                        f' at index {index_following_start}')

            index_following_end =\
                labels.index.get_indexer(end,
                                            method='bfill',
                                            tolerance=pd.Timedelta(5, unit='s'))[0]

            print('Found end timestamp within tolerance')
            print(f'Anomaly end: {end}')
            print(f'Timestamp within tolerance: '\
                        f'{index[index_following_end]}'\
                        f' at index {index_following_end}')

            column_indices = np.flatnonzero(process_labels == failure.Index)

            labels.iloc[index_following_start:index_following_end, column_indices] =\
                                                                int(failure.failure_source)

    return labels
            

def create_channel_names(median_labels, stdev_labels):

    median_labels = ['m_{}'.format(median_label)\
                        for median_label in median_labels]

    stdev_labels = ['std_{}'.format(stdev_label)
                        for stdev_label in stdev_labels]

    labels = np.concatenate((median_labels,
                                stdev_labels))

    return labels


def get_tpu_number(channel_name):
    parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
    # print(f'{channel_name}: {parameters}')
    return parameters[-1]


def get_process_label(channel_name):
    tpu_number = [int(substring) for substring in re.findall(r'\d+', channel_name)][-1]
    process_name = [substring for substring in re.findall(r'(?<=\|)[^:]+(?=:)', channel_name)][-1]

    if 'DCM' in process_name or\
            'MuCalReader' in process_name or\
            'NodeCoralProxy' in process_name or\
            'HLTMPPU' in process_name or\
            'HLTRC' in process_name:
        process_name = f'{process_name}_{tpu_number//1000}'

    # print(f'{tpu_number}|{process_name}')
    return f'{tpu_number}|{process_name}'


def fig_to_numpy_array(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    buf = np.array(fig.canvas.renderer.buffer_rgba())

    return cv.cvtColor(buf,cv.COLOR_RGBA2BGR)


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
                    labels_initial: np.array,
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

        self.labels = np.asarray(labels_initial)

        self.rack_numbers = [get_tpu_number(column) for column in self.columns]
    
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

        positions = (np.random.rand(math.ceil(self.dataset_length*ratio))*self.dataset_length).astype(int)

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

        positions = (np.random.rand(math.ceil(self.dataset_length*ratio))*\
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

        positions = (np.random.rand(math.ceil(self.dataset_length*ratio/\
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

        positions = (np.random.rand(math.ceil(self.dataset_length*ratio/\
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
            (np.random.rand(math.ceil(self.dataset_length*ratio/\
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
            (np.random.rand(math.ceil(self.dataset_length*ratio/\
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
            (np.random.rand(math.ceil(self.dataset_length*ratio_temporal/\
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

    parser = argparse.ArgumentParser(description='Real-World PMGPublishedProcessData Anomaly Dataset Generator')

    parser.add_argument('--variant', type=str)
    parser.add_argument('--dataset-dir', type=str, default='../../../../atlas-hlt-datasets')
    parser.add_argument('--generate-videos', action='store_true')
    parser.add_argument('--video-output-dir', type=str, default='../videos')
    
    args = parser.parse_args()

    # Load datasets

    train_set_x_df = pd.read_csv(f'{args.dataset_dir}/train_set_'\
                                    f'process_data_{args.variant}.csv', index_col=0)
    test_set_x_df = pd.read_csv(f'{args.dataset_dir}/test_set_'\
                                    f'process_data_{args.variant}.csv', index_col=0)
    val_set_x_df = pd.read_csv(f'{args.dataset_dir}/val_set_'\
                                    f'process_data_{args.variant}.csv', index_col=0)

    print(f'Train set size: {len(train_set_x_df)}')
    print(f'Test set size: {len(test_set_x_df)}')
    print(f'Val set size: {len(val_set_x_df)}')

    tpu_failure_log_df = pd.read_hdf(f'{args.dataset_dir}/'\
                                        f'tpu_failures_{args.variant}_'\
                                                'combined_preprocessed.h5')

    print(f'Anomaly count total: {len(tpu_failure_log_df)}')

    tpus_with_failures = np.array(list(set(
                            tpu_failure_log_df.index.get_level_values(1))))

    column_names_train = list((train_set_x_df).columns.values)
    column_names_test = list((test_set_x_df).columns.values)
    column_names_val = list((val_set_x_df).columns.values)

    print(f'Channels train: {len(column_names_train)}')
    print(f'Channels test: {len(column_names_test)}')
    print(f'Channels val: {len(column_names_val)}')

    intersection_train_test =\
                np.intersect1d(column_names_train,
                                    column_names_test)

    intersection_train_val =\
                np.intersect1d(column_names_train,
                                    column_names_val) 

    intersection_test_val =\
                np.intersect1d(column_names_test,
                                    column_names_val) 

    print(f'Train/test overlap: {len(intersection_train_test)}')
    print(f'Train/val overlap: {len(intersection_train_val)}')
    print(f'Test/val overlap: {len(intersection_test_val)}')

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

    process_labels_train = [get_process_label(label) for label in column_names_train]
    process_labels_test = [get_process_label(label) for label in column_names_test]
    process_labels_val = [get_process_label(label) for label in column_names_val]

    process_labels_train_unique = np.array(list(set(process_labels_train)))
    process_labels_test_unique = np.array(list(set(process_labels_test)))
    process_labels_val_unique = np.array(list(set(process_labels_val)))

    print(f'Unique TPUs train set: {len(process_labels_train_unique)}')
    print(f'Unique TPUs test set: {len(process_labels_test_unique)}')
    print(f'Unique TPUs val set: {len(process_labels_val_unique)}')

    exclusive_process_labels_train =\
        np.setdiff1d(process_labels_train_unique,
                        np.union1d(process_labels_test_unique,
                                        process_labels_val_unique))

    exclusive_process_labels_test =\
        np.setdiff1d(process_labels_test_unique,
                        np.union1d(process_labels_train_unique,
                                        process_labels_val_unique))

    exclusive_process_labels_val =\
        np.setdiff1d(process_labels_val_unique,
                        np.union1d(process_labels_train_unique,
                                        process_labels_test_unique))

    print(f'Train set unique labels:\n{exclusive_process_labels_train}')
    print(f'Test set unique labels:\n{exclusive_process_labels_test}')
    print(f'Val set unique labels:\n{exclusive_process_labels_val}')

    exclusive_tpus_with_failures_train =\
            np.intersect1d(exclusive_process_labels_train,
                                        tpus_with_failures)

    exclusive_tpus_with_failures_test =\
            np.intersect1d(exclusive_process_labels_test,
                                        tpus_with_failures)

    exclusive_tpus_with_failures_val =\
            np.intersect1d(exclusive_process_labels_val,
                                        tpus_with_failures)

    print(f'Exclusive TPUs with failures train:\n{exclusive_tpus_with_failures_train}')
    print(f'Exclusive TPUs with failures test:\n{exclusive_tpus_with_failures_test}')
    print(f'Exclusive TPUs with failures val:\n{exclusive_tpus_with_failures_val}')

    rack_numbers_train =\
        np.array([int(label.split('|')[0]) for label in process_labels_train])
    rack_numbers_test =\
        np.array([int(label.split('|')[0]) for label in process_labels_test])
    rack_numbers_val =\
        np.array([int(label.split('|')[0]) for label in process_labels_val])

    processes_train =\
        np.array([label.split('|')[-1] for label in process_labels_train])
    processes_test =\
        np.array([label.split('|')[-1] for label in process_labels_test])
    processes_val =\
        np.array([label.split('|')[-1] for label in process_labels_val])

    process_labels_train_unique, counts_train =\
        np.unique(processes_train, return_counts=True)

    print('Train set process counts:')

    for rack, count in zip(process_labels_train_unique, counts_train):
        print(f'{rack}: {count}')

    process_labels_test_unique, counts_test =\
        np.unique(processes_test, return_counts=True)

    print('Test set process counts:')

    for rack, count in zip(process_labels_test_unique, counts_test):
        print(f'{rack}: {count}')

    process_labels_val_unique, counts_val =\
        np.unique(processes_val, return_counts=True)

    print('Val set process counts:')

    for rack, count in zip(process_labels_val_unique, counts_val):
        print(f'{rack}: {count}') 

    # Unlabeled train set

    # Reduce dataset

    rack_data_train_unlabeled_all = []

    columns_reduced_train_unlabeled = None
    keys_last = None

    train_set_unlabeled_x_df = train_set_x_df

    print(f'Train set size total: {len(train_set_x_df)}')

    for count, row_x_data in enumerate(tqdm(train_set_unlabeled_x_df.to_numpy(),
                                                desc='Generating unlabeled train set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[processes_train[index]].append(datapoint)

        process_data_medians = {}
        process_data_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            process_data_medians[rack] = np.nanmedian(rack_bucket)
            process_data_stdevs[rack] = np.nanstd(rack_bucket)

        process_data_medians = dict(sorted(process_data_medians.items()))
        process_data_stdevs = dict(sorted(process_data_stdevs.items()))

        if keys_last != None:
            assert process_data_medians.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert process_data_medians.keys() == process_data_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = process_data_medians.keys()

        if type(columns_reduced_train_unlabeled) == type(None):
            columns_reduced_train_unlabeled = create_channel_names(process_data_medians.keys(),
                                                                    process_data_stdevs.keys())

        rack_data_np = np.concatenate((np.array(list(process_data_medians.values())),
                                            np.array(list(process_data_stdevs.values()))))

        rack_data_train_unlabeled_all.append(rack_data_np)

    rack_data_train_unlabeled_all_np = np.stack(rack_data_train_unlabeled_all)
    rack_data_train_unlabeled_all_np = np.nan_to_num(rack_data_train_unlabeled_all_np, nan=-1)

    nan_amount_train_unlabeled = 100*pd.isna(rack_data_train_unlabeled_all_np.flatten()).sum()/\
                                                            rack_data_train_unlabeled_all_np.size

    print('NaN amount reduced train set: {:.3f} %'.format(nan_amount_train_unlabeled))

    # Save dataset

    train_set_unlabeled_x_df = pd.DataFrame(rack_data_train_unlabeled_all_np,
                                                        train_set_unlabeled_x_df.index,
                                                        columns_reduced_train_unlabeled)

    train_set_unlabeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'\
                                            f'train_set_{args.variant}_x.h5',
                                        key='reduced_hlt_ppd_train_set_x',
                                        mode='w')

    if args.generate_videos:

        four_cc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_ppd_'\
                                            f'train_set_{args.variant}.mp4',
                                        four_cc, 60, (image_width, image_height))

        labels = rack_buckets_data.keys()

        for count in tqdm(range(len(rack_data_train_unlabeled_all_np)),
                            desc='Generating unlabeled train set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_train_unlabeled_all_np[lower_bound:count, :])\
                                if len(rack_data_train_unlabeled_all_np[lower_bound:count, :])\
                            else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            # ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Process Data")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Data")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_train_unlabeled_all_np[lower_bound:count, :])
            
            ax.legend(labels)

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Labeled train set

    test_set_size = len(test_set_x_df)

    train_set_labeled_x_df = test_set_x_df.iloc[:test_set_size//4, :]

    for count in range(1, len(train_set_labeled_x_df.index)):
        if train_set_labeled_x_df.index[count] <=\
                train_set_labeled_x_df.index[count-1]:
            print(f'Non-monotonic timestamp increase at {count-1}:\t'
                    f'First timestamp: {train_set_labeled_x_df.index[count-1]}\t'
                     f'Second timestamp: {train_set_labeled_x_df.index[count]}')

    column_names = train_set_labeled_x_df.columns
    timestamps = train_set_labeled_x_df.index

    # Generate labels for actual anomalies

    labels = generate_anomaly_labels(tpu_failure_log_df,
                                                timestamps,
                                                column_names,
                                                np.array(rack_numbers_test),
                                                prepad=0).to_numpy()
    
    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_train_labeled = MultivariateDataGenerator(train_set_labeled_x_df,
                                                                                labels,
                                                                                window_size_min=4,
                                                                                window_size_max=16)

    anomaly_generator_train_labeled.point_global_outliers(rack_count=1,
                                                                ratio=0.001,
                                                                factor=0.5,
                                                                radius=50)
    
    anomaly_generator_train_labeled.point_contextual_outliers(rack_count=1,
                                                                    ratio=0.001,
                                                                    factor=0.5,
                                                                    radius=50)

    anomaly_generator_train_labeled.persistent_global_outliers(rack_count=1,
                                                                    ratio=0.01,
                                                                    factor=1,
                                                                    radius=50)
    
    anomaly_generator_train_labeled.persistent_contextual_outliers(rack_count=1,
                                                                        ratio=0.005,
                                                                        factor=0.5,
                                                                        radius=50)

    anomaly_generator_train_labeled.collective_global_outliers(rack_count=1,
                                                                    ratio=0.005,
                                                                    option='square',
                                                                    coef=5,
                                                                    noise_amp=0.5,
                                                                    level=10,
                                                                    freq=0.1)

    anomaly_generator_train_labeled.collective_trend_outliers(rack_count=1,
                                                                    ratio=0.005,
                                                                    factor=0.5)

    # Reduce dataset and labels

    dataset = anomaly_generator_train_labeled.get_dataset_np()
    
    labels = remove_undetectable_anomalies(
                        np.nan_to_num(dataset),
                        anomaly_generator_train_labeled.get_labels_np())

    rack_data_train_labeled_all = []
    rack_labels_train_labeled_all = []

    columns_reduced_train_labeled = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(dataset, labels),
                                total=len(dataset),
                                desc='Generating labeled train set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[processes_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[processes_test[index]].append(label)

        process_data_medians = {}
        process_data_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            process_data_medians[rack] = np.nanmedian(rack_bucket)
            process_data_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        process_data_medians = dict(sorted(process_data_medians.items()))
        process_data_stdevs = dict(sorted(process_data_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert process_data_medians.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (process_data_medians.keys() == process_data_stdevs.keys()) and\
                                (process_data_medians.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = process_data_medians.keys()

        if type(columns_reduced_train_labeled) == type(None):
            columns_reduced_train_labeled = create_channel_names(process_data_medians.keys(),
                                                            process_data_stdevs.keys())
            
            assert np.array_equal(columns_reduced_train_labeled, columns_reduced_train_unlabeled),\
                                            "Labeled train columns don't match unlabeled train columns" 

        rack_data_np = np.concatenate((np.array(list(process_data_medians.values())),
                                            np.array(list(process_data_stdevs.values()))))

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
    
    # Save dataset and labels

    train_set_labeled_x_df = pd.DataFrame(rack_data_train_labeled_all_np,
                                                                timestamps,
                                                                columns_reduced_train_labeled)

    train_set_labeled_y_df = pd.DataFrame(rack_labels_train_labeled_all_np,
                                                                timestamps,
                                                                columns_reduced_train_labeled)

    anomalies_per_column = np.count_nonzero(rack_labels_train_labeled_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                len(rack_labels_train_labeled_all_np)

    for anomalies, anomaly_ratio, column_name in zip(anomalies_per_column,
                                                        anomaly_ratio_per_column,
                                                        columns_reduced_train_labeled):

        print(f'{column_name}: {anomalies} anomalies, {100*anomaly_ratio} % of all data')

    train_set_labeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'\
                                        f'labeled_train_set_{args.variant}_x.h5',
                                    key='reduced_hlt_ppd_labeled_train_set_x',
                                    mode='w')

    train_set_labeled_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'\
                                        f'labeled_train_set_{args.variant}_y.h5',
                                    key='reduced_hlt_ppd_labeled_train_set_y',
                                    mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_ppd_'
                                        f'labeled_train_set_{args.variant}.mp4',
                                    four_cc, 60, (image_width, image_height))


        for count in tqdm(range(len(rack_data_train_labeled_all_np)),
                        desc='Generating labeled train set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_train_labeled_all_np[lower_bound:count, :])\
                                if len(rack_data_train_labeled_all_np[lower_bound:count, :])\
                                else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_train_labeled_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Unreduced test set

    column_names = test_set_x_df.columns
    timestamps = test_set_x_df.index

    # Generate labels for actual anomalies

    labels_actual = generate_anomaly_labels(tpu_failure_log_df,
                                                    timestamps,
                                                    column_names,
                                                    np.array(rack_numbers_test),
                                                    prepad=0).to_numpy()

    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_test = MultivariateDataGenerator(test_set_x_df,
                                                            labels_actual,
                                                            window_size_min=4,
                                                            window_size_max=16)

    anomaly_generator_test.point_global_outliers(rack_count=1,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=8)
    
    anomaly_generator_test.point_contextual_outliers(rack_count=1,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=8)

    anomaly_generator_test.persistent_global_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            factor=0.5,
                                                            radius=8)
    
    anomaly_generator_test.persistent_contextual_outliers(rack_count=1,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=8)

    anomaly_generator_test.collective_global_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            option='square',
                                                            coef=5,
                                                            noise_amp=0.05,
                                                            level=10,
                                                            freq=0.1)

    anomaly_generator_test.collective_trend_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            factor=0.5)

    anomaly_generator_test.intra_rack_outliers(ratio_temporal=0.001,
                                                    ratio_channels=0.05,
                                                    average_duration=4.,
                                                    stdev_duration=1.)

    labels_unreduced =\
        remove_undetectable_anomalies(
            np.nan_to_num(anomaly_generator_test.get_dataset_np()),
            anomaly_generator_test.get_labels_np())

    # Save dataset and labels

    test_set_unreduced_x_df =\
        pd.DataFrame(anomaly_generator_test.get_dataset_np(),
                        anomaly_generator_test.get_timestamps_pd(),
                        test_set_x_df.columns)

    test_set_unreduced_y_df =\
        pd.DataFrame(labels_unreduced,
                        anomaly_generator_test.get_timestamps_pd(),
                        test_set_x_df.columns)

    test_set_unreduced_x_df.to_hdf(
            f'{args.dataset_dir}/unreduced_hlt_ppd_test_set_{args.variant}_x.h5',
            key='unreduced_hlt_ppd_test_set_x', mode='w')

    test_set_unreduced_y_df.to_hdf(
            f'{args.dataset_dir}/unreduced_hlt_ppd_test_set_{args.variant}_y.h5',
            key='unreduced__hlt_ppd_test_set_y', mode='w')
    
    # Reduced test set

    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_test = MultivariateDataGenerator(test_set_x_df,
                                                            labels_actual,
                                                            window_size_min=4,
                                                            window_size_max=16)

    anomaly_generator_test.point_global_outliers(rack_count=1,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=25)
    
    anomaly_generator_test.point_contextual_outliers(rack_count=1,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=25)

    anomaly_generator_test.persistent_global_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            factor=0.5,
                                                            radius=25)
    
    anomaly_generator_test.persistent_contextual_outliers(rack_count=1,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=25)

    anomaly_generator_test.collective_global_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            option='square',
                                                            coef=5,
                                                            noise_amp=0.05,
                                                            level=10,
                                                            freq=0.1)

    anomaly_generator_test.collective_trend_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            factor=0.5)

    # Reduce dataset and labels

    dataset = anomaly_generator_test.get_dataset_np()

    labels = remove_undetectable_anomalies(
                        np.nan_to_num(dataset),
                        anomaly_generator_test.get_labels_np())

    rack_data_test_all = []
    rack_labels_test_all = []

    columns_reduced_test = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(dataset, labels),
                                total=len(dataset),
                                desc='Generating test set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[processes_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[processes_test[index]].append(label)

        process_data_medians = {}
        process_data_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            process_data_medians[rack] = np.nanmedian(rack_bucket)
            process_data_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        process_data_medians = dict(sorted(process_data_medians.items()))
        process_data_stdevs = dict(sorted(process_data_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert process_data_medians.keys() == keys_last,\
                            'Rack bucket keys changed between slices'

            assert (process_data_medians.keys() == process_data_stdevs.keys()) and\
                                (process_data_medians.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = process_data_medians.keys()

        if type(columns_reduced_test) == type(None):
            columns_reduced_test = create_channel_names(process_data_medians.keys(),
                                                            process_data_stdevs.keys())
            
            assert np.array_equal(columns_reduced_test, columns_reduced_train_unlabeled),\
                                                    "Test columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(process_data_medians.values())),
                                            np.array(list(process_data_stdevs.values()))))

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
    
    # Save dataset and labels

    test_set_reduced_x_df = pd.DataFrame(rack_data_test_all_np,
                                            anomaly_generator_test.get_timestamps_pd(),
                                            columns_reduced_test)

    test_set_reduced_y_df = pd.DataFrame(rack_labels_test_all_np,
                                            anomaly_generator_test.get_timestamps_pd(),
                                            columns_reduced_test)

    anomalies_per_column = np.count_nonzero(rack_labels_test_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                    len(rack_labels_test_all_np)

    for anomalies, anomaly_ratio, column_name in zip(anomalies_per_column,
                                                        anomaly_ratio_per_column,
                                                        columns_reduced_test):

        print(f'{column_name}: {anomalies} anomalies, {100*anomaly_ratio} % of all data')

    test_set_reduced_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'
                                        f'test_set_{args.variant}_x.h5',
                                    key='reduced_hlt_ppd_test_set_x',
                                    mode='w')

    test_set_reduced_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'
                                        f'test_set_{args.variant}_y.h5',
                                    key='reduced_hlt_ppd_test_set_y',
                                    mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_ppd_'
                                        f'test_set_{args.variant}.mp4',
                                    four_cc, 60, (image_width, image_height))

        for count in tqdm(range(len(rack_data_test_all_np)),
                                    desc='Generating test set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_test_all_np[lower_bound:count, :])\
                                if len(rack_data_test_all_np[lower_bound:count, :])\
                                else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            # ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_test_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Clean val set

    # Reduce dataset

    clean_val_set_x_df = val_set_x_df

    column_names = clean_val_set_x_df.columns
    timestamps = clean_val_set_x_df.index

    rack_data_clean_val_all = []

    columns_reduced_clean_val = None
    keys_last = None

    for count, row_x_data in enumerate(tqdm(val_set_x_df.to_numpy(),
                                                desc='Generating clean val set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[processes_val[index]].append(datapoint)

        process_data_medians = {}
        process_data_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            process_data_medians[rack] = np.nanmedian(rack_bucket)
            process_data_stdevs[rack] = np.nanstd(rack_bucket)

        process_data_medians = dict(sorted(process_data_medians.items()))
        process_data_stdevs = dict(sorted(process_data_stdevs.items()))

        if keys_last != None:
            assert process_data_medians.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert process_data_medians.keys() == process_data_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = process_data_medians.keys()

        if type(columns_reduced_clean_val) == type(None):
            columns_reduced_clean_val = create_channel_names(process_data_medians.keys(),
                                                                process_data_stdevs.keys())

            assert np.array_equal(columns_reduced_clean_val, columns_reduced_train_unlabeled),\
                                                        "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(process_data_medians.values())),
                                            np.array(list(process_data_stdevs.values()))))

        rack_data_clean_val_all.append(rack_data_np)

    rack_data_clean_val_all_np = np.stack(rack_data_clean_val_all)
    rack_data_clean_val_all_np = np.nan_to_num(rack_data_clean_val_all_np, nan=-1)

    nan_amount_clean_val = 100*pd.isna(rack_data_clean_val_all_np.flatten()).sum()/\
                                                    rack_data_clean_val_all_np.size

    print('NaN amount reduced clean val set: {:.3f} %'.format(nan_amount_clean_val))

    # Save dataset

    clean_val_set_x_df = pd.DataFrame(rack_data_clean_val_all_np,
                                                val_set_x_df.index,
                                                columns_reduced_clean_val)

    clean_val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'
                                    f'clean_val_set_{args.variant}_x.h5',
                                key='reduced_hlt_ppd_clean_val_set_x',
                                mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_ppd_'
                                        f'clean_val_set_{args.variant}.mp4',
                                    four_cc, 60, (image_width, image_height))

        for count in tqdm(range(len(rack_data_clean_val_all_np)),
                                    desc='Generating clean val set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_clean_val_all_np[lower_bound:count, :])\
                                    if len(rack_data_clean_val_all_np[lower_bound:count, :])\
                                    else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            # ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_clean_val_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Dirty val set

    val_set_x_df = pd.concat((val_set_x_df.iloc[:776, :],
                                test_set_x_df.iloc[-716:, :]))

    column_names_val = list((val_set_x_df).columns.values)
    process_labels_val = [get_process_label(label) for label in column_names_val]

    rack_numbers_val =\
        np.array([int(label.split('|')[0]) for label in process_labels_val])

    processes_val =\
        np.array([label.split('|')[-1] for label in process_labels_val])

    for count in range(1, len(val_set_x_df.index)):
        if val_set_x_df.index[count] <=\
                val_set_x_df.index[count-1]:
            print(f'Non-monotonic timestamp increase at {count-1}:\t'
                    f'First timestamp: {val_set_x_df.index[count-1]}\t'
                     f'Second timestamp: {val_set_x_df.index[count]}')

    column_names = val_set_x_df.columns
    timestamps = val_set_x_df.index

    # Generate labels for actual anomalies

    labels_actual = generate_anomaly_labels(tpu_failure_log_df,
                                                    timestamps,
                                                    column_names,
                                                    np.array(rack_numbers_val),
                                                    prepad=0).to_numpy()
    
    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_val = MultivariateDataGenerator(val_set_x_df,
                                                        labels_actual,
                                                        window_size_min=4,
                                                        window_size_max=16)

    anomaly_generator_val.point_global_outliers(rack_count=1,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=25)
    
    anomaly_generator_val.point_contextual_outliers(rack_count=1,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=25)

    anomaly_generator_val.persistent_global_outliers(rack_count=1,
                                                            ratio=0.005,
                                                            factor=0.5,
                                                            radius=25)
    
    anomaly_generator_val.persistent_contextual_outliers(rack_count=1,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=25)

    anomaly_generator_val.collective_global_outliers(rack_count=1,
                                                        ratio=0.005,
                                                        option='square',
                                                        coef=5,
                                                        noise_amp=0.5,
                                                        level=10,
                                                        freq=0.1)

    anomaly_generator_val.collective_trend_outliers(rack_count=1,
                                                        ratio=0.005,
                                                        factor=0.5)
     
    # Reduce dataset and labels
    
    dataset = anomaly_generator_val.get_dataset_np()

    labels = remove_undetectable_anomalies(
                        np.nan_to_num(dataset),
                        anomaly_generator_val.get_labels_np())

    rack_data_val_all = []
    rack_labels_val_all = []

    columns_reduced_val = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(dataset, labels),
                                    total=len(dataset),
                                    desc='Generating dirty val set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[processes_val[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[processes_val[index]].append(label)

        process_data_medians = {}
        process_data_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            process_data_medians[rack] = np.nanmedian(rack_bucket)
            process_data_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        process_data_medians = dict(sorted(process_data_medians.items()))
        process_data_stdevs = dict(sorted(process_data_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert process_data_medians.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (process_data_medians.keys() == process_data_stdevs.keys()) and\
                                (process_data_medians.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = process_data_medians.keys()

        if type(columns_reduced_val) == type(None):
            columns_reduced_val = create_channel_names(process_data_medians.keys(),
                                                        process_data_stdevs.keys())

            print(columns_reduced_train_unlabeled)
            print(columns_reduced_val)

            assert np.array_equal(columns_reduced_val, columns_reduced_train_unlabeled),\
                                                    "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(process_data_medians.values())),
                                            np.array(list(process_data_stdevs.values()))))

        rack_data_val_all.append(rack_data_np)

        rack_labels_val_all.append(np.array(list(rack_labels.values())))

    rack_data_val_all_np = np.stack(rack_data_val_all)
    rack_data_val_all_np = np.nan_to_num(rack_data_val_all_np, nan=-1)

    nan_amount_dirty_val = 100*pd.isna(rack_data_val_all_np.flatten()).sum()/\
                                                    rack_data_val_all_np.size

    print('NaN amount reduced dirty val set: {:.3f} %'.format(nan_amount_dirty_val))

    rack_labels_val_all_np = np.stack(rack_labels_val_all)

    rack_labels_val_all_np = np.concatenate([rack_labels_val_all_np,\
                                                rack_labels_val_all_np],
                                                axis=1)

    val_set_x_df = pd.DataFrame(rack_data_val_all_np,
                                    anomaly_generator_val.get_timestamps_pd(),
                                    columns_reduced_val)

    val_set_y_df = pd.DataFrame(rack_labels_val_all_np,
                                    anomaly_generator_val.get_timestamps_pd(),
                                    columns_reduced_val)

    anomalies_per_column = np.count_nonzero(rack_labels_val_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                    len(rack_labels_val_all_np)
    
    # Save dataset and labels

    val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'
                                f'val_set_{args.variant}_x.h5',
                            key='reduced_hlt_ppd_val_set_x',
                            mode='w')

    val_set_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_ppd_'
                                f'val_set_{args.variant}_y.h5',
                            key='reduced_hlt_ppd_val_set_y',
                            mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_ppd_'
                                            f'val_set_{args.variant}.mp4',
                                    four_cc, 60,(image_width, image_height))

        for count in tqdm(range(len(rack_data_val_all_np)),
                            desc='Generating dirty val set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_val_all_np[lower_bound:count, :])\
                                if len(rack_data_val_all_np[lower_bound:count, :])\
                                else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            # ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_val_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()


