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

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

run_endpoints = [1404,
                    8928,
                    19296,
                    28948]

channels_to_delete_last_run = [1357,
                                3685,
                                3184]

rack_colors = {  0: '#D81B60',
                    1: '#1E88E5',
                    2: '#FFC107',
                    3: '#004D40',
                    4: '#C43F42',
                    5: '#6F8098',
                    6: '#D4FC14',
                    7: '#1CB2C5',
                    8: '#18F964',
                    9: '#1164B3'}

SMALL_SIZE = 13
MEDIUM_SIZE = 13
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def get_filename(path: str):
    filename_with_extension = os.path.basename(path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension


def get_tpu_number(channel_name):
    parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
    return parameters[4]


def get_rack_hardware_configuration(rack_number: int,
                                        variant: str = '2018'):
    
    if variant == '2018':

        if 44 <= rack_number <= 54:
            return 0
        elif 55 <= rack_number <= 63:
            return 1
        elif (70 <= rack_number <= 77) or\
                    (79 <= rack_number <= 90):
            return 2
        elif 16 <= rack_number <= 26:
            return 3
        else:
            raise ValueError(f'Rack number {rack_number} not '
                                f'in known nodes for variant {variant}')

    else:
        raise NotImplementedError('Rack hardware configuration '
                                    'identification not implemented '
                                    f'for variant {variant}')


if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='HLT Dataset Plot Generator')

    parser.add_argument('--dataset', type=str, default='../datasets/hlt/hlt_train_set_2018.csv')
    parser.add_argument('--variant', type=str, default='2018')
    parser.add_argument('--lower-bound', type=int, default=0)
    parser.add_argument('--upper-bound', type=int, default=-1)
    
    args = parser.parse_args()

    dataset_df = pd.read_csv(args.dataset, index_col=0)

    if args.lower_bound < 0:
        raise ValueError('Lower bound cannot be less than 0')

    xlim_lower = args.lower_bound

    if args.upper_bound > len(dataset_df):
        raise ValueError(f'Upper bound {args.upper_bound} '
                            'out of range for dataset '
                            f'of length {len(dataset_df)}')
    else:
        xlim_upper = args.upper_bound
    
    column_names = list(dataset_df.columns.values)

    print(f'Channels: {len(column_names)}')

    nan_amount = np.mean(np.sum(pd.isna(dataset_df.to_numpy()), 1)/dataset_df.shape[1])

    print(f'Mean sparsity original dataset: {100*nan_amount:.3f} %')

    dataset_df.dropna(axis=0,
                        thresh=50,
                        inplace=True)

    dataset_np = dataset_df.to_numpy()

    nan_amount = np.mean(np.sum(pd.isna(dataset_np), 1)/dataset_df.shape[1])

    print(f'Mean sparsity preprocessed: {100*nan_amount:.3f} %')

    tpu_numbers = [get_tpu_number(label) for label in column_names]
    tpu_numbers_unique = np.array(list(set(tpu_numbers)))
    rack_numbers = np.floor_divide(tpu_numbers, 1000)

    hardware_configurations =\
        [get_rack_hardware_configuration(rack_number, args.variant) + 1\
                                        for rack_number in rack_numbers]

    channel_colors = [rack_colors[configuration]\
                            for configuration in hardware_configurations]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # plt.yticks(rotation=30, ha='right')

    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Event Rate [Hz]')

    ax.set_ylim(-1, 80)

    ax.grid()

    # x = np.arange(len(dataset_np[xlim_lower:xlim_upper, :]), dtype=np.int64)*5
    x = np.arange(len(dataset_np[xlim_lower:xlim_upper, :]))*5/3600

    # for channel in trange(dataset_np.shape[-1] - 1, 0, -1, desc='Plotting'): 
    for channel in trange(dataset_np.shape[-1], desc='Plotting'):
        ax.plot(x, dataset_np[xlim_lower:xlim_upper, channel],
                    linewidth=1, color=channel_colors[channel])

    dataset_name = get_filename(args.dataset)

    plt.tight_layout()

    plt.savefig(f'plot_{dataset_name}.png')
