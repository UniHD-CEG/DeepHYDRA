#!/usr/bin/env python3
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
                                        tpu_numbers: np.array,
                                        prepad: int = 0):

    index = pd.DatetimeIndex(index)

#     print(index[0])
#     find_timestamp_jumps(index)
#     print(index[-1])
    
    labels = pd.DataFrame(0, index, columns, dtype=np.uint32)

    # print(failure_data)

    # exit()

    failure_data = failure_data.droplevel(0)

    for failure in failure_data.itertuples():

        start = pd.DatetimeIndex([failure.start], tz=index.tz)
        end = pd.DatetimeIndex([failure.end], tz=index.tz)
        
        # print(f'{failure.Index}: ', end='')
        # print(f'{failure.Index}: ')

        # if np.any(index >= start[0]) and\
        #             np.any(index <= end[0]):
        #     print(f'Got anomaly {int(failure.failure_source)}'\
        #                         f' between {start[0]} and {end[0]}')

        # else:
        #     print(f'No anomalies between {start[0]} and {end[0]}')

        # Check if any timestamp in the index is
        # close to the start of the anomaly

        # print(labels.index.get_indexer(start,
        #                                 method='bfill',
        #                                 tolerance=pd.Timedelta(5, unit='m')))

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
            
            # print(labels.index.get_indexer(end,
            #                                 method='bfill',
            #                                 tolerance=pd.Timedelta(5, unit='s')))


            index_following_end =\
                labels.index.get_indexer(end,
                                            method='bfill',
                                            tolerance=pd.Timedelta(5, unit='s'))[0]


            print('Found end timestamp within tolerance')
            print(f'Anomaly end: {end}')
            print(f'Timestamp within tolerance: '\
                        f'{index[index_following_end]}'\
                        f' at index {index_following_end}')

            column_indices = np.flatnonzero(tpu_numbers == failure.Index)

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


if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Real-World DCM Rate Anomaly Dataset Generator')

    parser.add_argument('--variant', type=str)
    parser.add_argument('--dataset-dir', type=str, default='../../../../atlas-hlt-datasets')
    parser.add_argument('--generate-videos', action='store_true')
    parser.add_argument('--video-output-dir', type=str, default='../videos')
    
    args = parser.parse_args()

    # Load datasets

    train_set_x_df = pd.read_csv(f'{args.dataset_dir}/train_set_'\
                                    f'dcm_rates_{args.variant}.csv', index_col=0)
    test_set_x_df = pd.read_csv(f'{args.dataset_dir}/test_set_'\
                                    f'dcm_rates_{args.variant}.csv', index_col=0)
    val_set_x_df = pd.read_csv(f'{args.dataset_dir}/val_set_'\
                                    f'dcm_rates_{args.variant}.csv', index_col=0)

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

#     print('Train channel names')
# 
#     for name in column_names_train:
#         print(name, end=' ')
# 
#     print('Test channel names')
# 
#     for name in column_names_test:
#         print(name, end=' ')
# 
#     print('Val channel names')
# 
#     for name in column_names_val:
#         print(name, end=' ')
# 
#     exit()

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

    def get_tpu_number(channel_name):
        parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
        # print(f'{channel_name}: {parameters}')
        return parameters[-1]

    tpu_numbers_train = [get_tpu_number(label) for label in column_names_train]
    tpu_numbers_test = [get_tpu_number(label) for label in column_names_test]
    tpu_numbers_val = [get_tpu_number(label) for label in column_names_val]

    tpu_numbers_train_unique = np.array(list(set(tpu_numbers_train)))
    tpu_numbers_test_unique = np.array(list(set(tpu_numbers_test)))
    tpu_numbers_val_unique = np.array(list(set(tpu_numbers_val)))

    rack_numbers_train = np.floor_divide(tpu_numbers_train, 1000)
    rack_numbers_test = np.floor_divide(tpu_numbers_test, 1000)
    rack_numbers_val = np.floor_divide(tpu_numbers_val, 1000)

#     print('Train rack numbers')
# 
#     for name in rack_numbers_train:
#         print(name, end=' ')
# 
#     print('Test rack numbers')
# 
#     for name in rack_numbers_test:
#         print(name, end=' ')
# 
#     print('Val rack numbers')
# 
#     for name in rack_numbers_val:
#         print(name, end=' ')

    racks_train, counts_train =\
        np.unique(rack_numbers_train, return_counts=True)

    print('Train set TPUs per rack:')

    for rack, count in zip(racks_train, counts_train):
        print(f'{rack}: {count}')

    racks_test, counts_test =\
        np.unique(rack_numbers_test, return_counts=True)

    print('Test set TPUs per rack:')

    for rack, count in zip(racks_test, counts_test):
        print(f'{rack}: {count}')

    racks_val, counts_val =\
        np.unique(rack_numbers_val, return_counts=True)

    print('Val set TPUs per rack:')

    for rack, count in zip(racks_val, counts_val):
        print(f'{rack}: {count}') 

    print(f'Unique TPUs train set: {len(tpu_numbers_train_unique)}')
    print(f'Unique TPUs test set: {len(tpu_numbers_test_unique)}')
    print(f'Unique TPUs val set: {len(tpu_numbers_val_unique)}')

    exclusive_tpu_numbers_train =\
        np.setdiff1d(tpu_numbers_train_unique,
                        np.union1d(tpu_numbers_test_unique,
                                        tpu_numbers_val_unique))

    exclusive_tpu_numbers_test =\
        np.setdiff1d(tpu_numbers_test_unique,
                        np.union1d(tpu_numbers_train_unique,
                                        tpu_numbers_val_unique))

    exclusive_tpu_numbers_val =\
        np.setdiff1d(tpu_numbers_val_unique,
                        np.union1d(tpu_numbers_train_unique,
                                        tpu_numbers_test_unique))

    print(f'Train set unique TPUs:\n{exclusive_tpu_numbers_train}')
    print(f'Test set unique TPUs:\n{exclusive_tpu_numbers_test}')
    print(f'Val set unique TPUs:\n{exclusive_tpu_numbers_val}')

    exclusive_tpus_with_failures_train =\
            np.intersect1d(exclusive_tpu_numbers_train,
                                        tpus_with_failures)

    exclusive_tpus_with_failures_test =\
            np.intersect1d(exclusive_tpu_numbers_test,
                                        tpus_with_failures)

    exclusive_tpus_with_failures_val =\
            np.intersect1d(exclusive_tpu_numbers_val,
                                        tpus_with_failures)

    print(f'Exclusive TPUs with failures train:\n{exclusive_tpus_with_failures_train}')
    print(f'Exclusive TPUs with failures test:\n{exclusive_tpus_with_failures_test}')
    print(f'Exclusive TPUs with failures val:\n{exclusive_tpus_with_failures_val}')

    # Reduce and save unlabened train set

    rack_data_train_unlabeled_all = []

    columns_reduced_train_unlabeled = None
    keys_last = None

    # train_set_size = len(train_set_x_df)

    # train_set_unlabeled_x_df = train_set_x_df.iloc[:4*train_set_size//5, :]
    train_set_unlabeled_x_df = train_set_x_df

    print(f'Train set size total: {len(train_set_x_df)}')

    for count, row_x_data in enumerate(tqdm(train_set_unlabeled_x_df.to_numpy(),
                                                desc='Generating unlabeled train set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_train[index]].append(datapoint)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_train_unlabeled) == type(None):
            columns_reduced_train_unlabeled = create_channel_names(rack_median_dcm_rates.keys(),
                                                                    rack_dcm_rate_stdevs.keys())

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_train_unlabeled_all.append(rack_data_np)

    rack_data_train_unlabeled_all_np = np.stack(rack_data_train_unlabeled_all)
    rack_data_train_unlabeled_all_np = np.nan_to_num(rack_data_train_unlabeled_all_np, nan=-1)

    nan_amount_train_unlabeled = 100*pd.isna(rack_data_train_unlabeled_all_np.flatten()).sum()/\
                                                            rack_data_train_unlabeled_all_np.size

    print('NaN amount reduced train set: {:.3f} %'.format(nan_amount_train_unlabeled))

    train_set_unlabeled_x_df = pd.DataFrame(rack_data_train_unlabeled_all_np,
                                                        train_set_unlabeled_x_df.index,
                                                        columns_reduced_train_unlabeled)

    train_set_unlabeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'\
                                            f'train_set_{args.variant}_x.h5',
                                        key='reduced_hlt_train_set_x',
                                        mode='w')

    if args.generate_videos:

        four_cc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'\
                                            f'train_set_{args.variant}.mp4',
                                        four_cc, 60, (image_width, image_height))

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
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_train_unlabeled_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Reduce and save labeled train set

    test_set_size = len(test_set_x_df)

    train_set_labeled_x_df = test_set_x_df.iloc[:test_set_size//4, :]

    for count in range(1, len(train_set_labeled_x_df.index)):
        if train_set_labeled_x_df.index[count] <=\
                train_set_labeled_x_df.index[count-1]:
            print(f'Non-monotonic timestamp increase at {count-1}:\t'
                    f'First timestamp: {train_set_labeled_x_df.index[count-1]}\t'
                     f'Second timestamp: {train_set_labeled_x_df.index[count]}')

    dataset = train_set_labeled_x_df.to_numpy()
    column_names = train_set_labeled_x_df.columns
    timestamps = train_set_labeled_x_df.index

    labels = generate_anomaly_labels(tpu_failure_log_df,
                                                timestamps,
                                                column_names,
                                                np.array(tpu_numbers_test),
                                                prepad=5).to_numpy()

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
            rack_buckets_data[rack_numbers_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_test[index]].append(label)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys()) and\
                                (rack_median_dcm_rates.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_train_labeled) == type(None):
            columns_reduced_train_labeled = create_channel_names(rack_median_dcm_rates.keys(),
                                                            rack_dcm_rate_stdevs.keys())
            
            assert np.array_equal(columns_reduced_train_labeled, columns_reduced_train_unlabeled),\
                                            "Labeled train columns don't match unlabeled train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

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

    train_set_labeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'\
                                        f'labeled_train_set_{args.variant}_x.h5',
                                    key='reduced_hlt_labeled_train_set_x',
                                    mode='w')

    train_set_labeled_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'\
                                        f'labeled_train_set_{args.variant}_y.h5',
                                    key='reduced_hlt_labeled_train_set_y',
                                    mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
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


    # Reduce and save test set

    # Save unreduced test set for testing of combined DBSCAN/Transformer-based
    # detection pipeline

    column_names = test_set_x_df.columns
    timestamps = test_set_x_df.index

    test_set_y_df = generate_anomaly_labels(tpu_failure_log_df,
                                                    timestamps,
                                                    column_names,
                                                    np.array(tpu_numbers_test),
                                                    prepad=5)

    test_set_x_df.to_hdf(f'{args.dataset_dir}/unreduced_hlt_test_set_{args.variant}_x.h5',
                                                key='unreduced_hlt_test_set_x', mode='w')

    test_set_y_df.to_hdf(f'{args.dataset_dir}/unreduced_hlt_test_set_{args.variant}_y.h5',
                                                    key='reduced_hlt_test_set_y', mode='w')

    rack_data_test_all = []
    rack_labels_test_all = []

    columns_reduced_test = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(test_set_x_df.to_numpy(),
                                    test_set_y_df.to_numpy()),
                                total=len(test_set_x_df),
                                desc='Generating test set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_test[index]].append(label)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                            'Rack bucket keys changed between slices'

            assert (rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys()) and\
                                (rack_median_dcm_rates.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_test) == type(None):
            columns_reduced_test = create_channel_names(rack_median_dcm_rates.keys(),
                                                            rack_dcm_rate_stdevs.keys())
            
            assert np.array_equal(columns_reduced_test, columns_reduced_train_unlabeled),\
                                                    "Test columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

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

    test_set_reduced_x_df = pd.DataFrame(rack_data_test_all_np,
                                                test_set_x_df.index,
                                                columns_reduced_test)

    test_set_reduced_y_df = pd.DataFrame(rack_labels_test_all_np,
                                                test_set_y_df.index,
                                                columns_reduced_test)

    anomalies_per_column = np.count_nonzero(rack_labels_test_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                    len(rack_labels_test_all_np)

    for anomalies, anomaly_ratio, column_name in zip(anomalies_per_column,
                                                        anomaly_ratio_per_column,
                                                        columns_reduced_test):

        print(f'{column_name}: {anomalies} anomalies, {100*anomaly_ratio} % of all data')

    test_set_reduced_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                        f'test_set_{args.variant}_x.h5',
                                    key='reduced_hlt_test_set_x',
                                    mode='w')

    test_set_reduced_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                        f'test_set_{args.variant}_y.h5',
                                    key='reduced_hlt_test_set_y',
                                    mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
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
            ax.set_ylim(-2, max_val_slice + 10)

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

    # Reduce and save clean val set

    # clean_val_set_x_df = pd.concat((val_set_x_df.iloc[:10500, :],
    #                             val_set_x_df.iloc[11500:14500, :]))

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
            rack_buckets_data[rack_numbers_val[index]].append(datapoint)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_clean_val) == type(None):
            columns_reduced_clean_val = create_channel_names(rack_median_dcm_rates.keys(),
                                                                rack_dcm_rate_stdevs.keys())

            assert np.array_equal(columns_reduced_clean_val, columns_reduced_train_unlabeled),\
                                                        "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_clean_val_all.append(rack_data_np)

    rack_data_clean_val_all_np = np.stack(rack_data_clean_val_all)
    rack_data_clean_val_all_np = np.nan_to_num(rack_data_clean_val_all_np, nan=-1)

    nan_amount_clean_val = 100*pd.isna(rack_data_clean_val_all_np.flatten()).sum()/\
                                                    rack_data_clean_val_all_np.size

    print('NaN amount reduced clean val set: {:.3f} %'.format(nan_amount_clean_val))

    clean_val_set_x_df = pd.DataFrame(rack_data_clean_val_all_np,
                                                val_set_x_df.index,
                                                columns_reduced_clean_val)

    clean_val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                    f'clean_val_set_{args.variant}_x.h5',
                                key='reduced_hlt_clean_val_set_x',
                                mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
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
            ax.set_ylim(-2, max_val_slice + 10)

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

    # Reduce and save dirty val set

    val_set_x_df = pd.concat((val_set_x_df.iloc[:9270, :],
                                test_set_x_df.iloc[-8570:, :]))

    column_names_val = list((val_set_x_df).columns.values)
    tpu_numbers_val = [get_tpu_number(label) for label in column_names_val]
    tpu_numbers_val_unique = np.array(list(set(tpu_numbers_val)))
    rack_numbers_val = np.floor_divide(tpu_numbers_val, 1000)

    for count in range(1, len(val_set_x_df.index)):
        if val_set_x_df.index[count] <=\
                val_set_x_df.index[count-1]:
            print(f'Non-monotonic timestamp increase at {count-1}:\t'
                    f'First timestamp: {val_set_x_df.index[count-1]}\t'
                     f'Second timestamp: {val_set_x_df.index[count]}')

    column_names = val_set_x_df.columns
    timestamps = val_set_x_df.index

    labels = generate_anomaly_labels(tpu_failure_log_df,
                                                timestamps,
                                                column_names,
                                                np.array(tpu_numbers_test),
                                                prepad=5).to_numpy()

    rack_data_val_all = []
    rack_labels_val_all = []

    columns_reduced_val = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(val_set_x_df.to_numpy(), labels),
                                            total=len(val_set_x_df),
                                            desc='Generating dirty val set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_val[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_val[index]].append(label)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys()) and\
                                (rack_median_dcm_rates.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_val) == type(None):
            columns_reduced_val = create_channel_names(rack_median_dcm_rates.keys(),
                                                        rack_dcm_rate_stdevs.keys())

            assert np.array_equal(columns_reduced_val, columns_reduced_train_unlabeled),\
                                                    "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

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
                                    val_set_x_df.index,
                                    columns_reduced_val)

    val_set_y_df = pd.DataFrame(rack_labels_val_all_np,
                                    val_set_x_df.index,
                                    columns_reduced_val)

    anomalies_per_column = np.count_nonzero(rack_labels_val_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                    len(rack_labels_val_all_np)

    val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                f'val_set_{args.variant}_x.h5',
                            key='reduced_hlt_val_set_x',
                            mode='w')

    val_set_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                f'val_set_{args.variant}_y.h5',
                            key='reduced_hlt_val_set_y',
                            mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
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
            ax.set_ylim(-2, max_val_slice + 10)

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


