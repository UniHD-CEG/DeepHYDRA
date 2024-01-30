import os
from multiprocessing import Pool
from functools import partial
import argparse

import numpy as np
import bottleneck as bn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,\
                                f1_score,\
                                precision_score,\
                                recall_score,\
                                matthews_corrcoef

from tqdm.auto import trange
# import pylikwid

run_endpoints = [1404,
                    8928,
                    19296,
                    28948]

channels_to_delete_last_run = [1357,
                                3685,
                                3184]


def save_numpy_array(array: np.array,
                        filename: str):    
    with open(filename, 'wb') as output_file:
        np.save(output_file, array)


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def method_4_fixed_k(k: int,
                        abs_diff: np.array,
                        labels: np.array,
                        b_array: np.array,
                        c_array: np.array) -> np.array:

    moving_mean = bn.move_mean(abs_diff,
                                    window=k,
                                    axis=0)
    
    moving_mean = np.nan_to_num(moving_mean)
    
    moving_std = bn.move_std(abs_diff,
                                window=k,
                                axis=0)
    
    moving_std = np.nan_to_num(moving_std)

    channel_count = abs_diff.shape[-1]

    mcc_best_per_channel = np.full(channel_count, -1.)
    b_best_per_channel = np.full(channel_count, -1.)
    c_best_per_channel = np.full(channel_count, -1.)

    configuration_count = len(c_array)*len(b_array)

    for count_outer, c in enumerate(c_array):
        for count_inner, b in enumerate(b_array):
            for channel in range(channel_count):

                abs_diff_channel = abs_diff[:, channel]
                moving_mean_channel = moving_mean[:, channel]
                moving_std_channel = moving_std[:, channel]

                labels_channel = labels[:, channel]

                preds_channel = np.greater(abs_diff_channel,
                                            moving_mean_channel*\
                                            c*moving_std_channel + b)\
                                                        .astype(np.uint8)

                preds_channel =\
                    adjust_predicts(preds_channel,
                                        labels_channel, 0.1)
        
                mcc = matthews_corrcoef(labels_channel,
                                            preds_channel)

                if mcc_best_per_channel[channel] < mcc:

                    print(f'Channel {channel} new best MCC={mcc:.3f}, '
                                        f'{b=:.3f}, {c=:.3f}, {k=:.3f}')

                    mcc_best_per_channel[channel] = mcc
                    b_best_per_channel[channel] = b
                    c_best_per_channel[channel] = c

            percent_done = 100*(count_inner + 1 +\
                                count_outer*len(b_array))/\
                                                configuration_count

            print(f'{k=}: {percent_done:.2f} % done')

    return k, mcc_best_per_channel,\
                    b_best_per_channel,\
                    c_best_per_channel


def method_4_combined(abs_diff: np.array,
                            labels: np.array,
                            parameters: np.array) -> np.array:

    channel_count = abs_diff.shape[-1]

    preds_all = np.zeros_like(abs_diff)

    for channel in trange(channel_count):

        b = parameters[channel, 0]
        c = parameters[channel, 1]
        k = int(parameters[channel, 2])

        abs_diff_channel = abs_diff[:, channel]

        moving_mean_channel =\
            bn.move_mean(abs_diff_channel,
                                    window=k,
                                    axis=0)

        moving_std_channel =\
                bn.move_std(abs_diff_channel,
                                    window=k,
                                    axis=0)

        labels_channel = labels[:, channel]

        preds_channel = np.greater(abs_diff_channel,
                                    moving_mean_channel*\
                                    c*moving_std_channel + b)\
                                                .astype(np.uint8)

        preds_all[:, channel] =\
                adjust_predicts(preds_channel,
                                    labels_channel, 0.1)

    preds_all = np.nan_to_num(preds_all)

    return preds_all


def parameter_exploration(data: np.array,
                            labels: np.array,
                            b_lower: np.float64=0.,
                            b_upper: np.float64=1.,
                            b_count: int=16,
                            c_lower: np.float64=0.,
                            c_upper: np.float64=1.,
                            c_count: int=16,
                            k_lower: int=2,
                            k_upper: int=16):

    '''
    One-liner-based anomaly detection as defined by
    R. Wu and E. J. Keogh in "Current Time Series Anomaly
    Detection Benchmarks are Flawed and are Creating
    the Illusion of Progress (Extended Abstract),"
    2022 IEEE 38th International Conference on Data Engineering
    (ICDE), Kuala Lumpur, Malaysia, 2022, pp. 1479-1480,
    doi: 10.1109/ICDE53745.2022.00116.
    '''

    diff = np.diff(data, axis=0)

    labels = labels[1:, :]

    b_array = np.linspace(b_lower, b_upper, b_count)
    c_array = np.linspace(c_lower, c_upper, c_count)
    k_array = np.arange(k_lower, k_upper + 1)

    abs_diff = np.abs(diff)

    abs_diff_normalized = MinMaxScaler().fit_transform(abs_diff)

    channel_count = diff.shape[-1]

    auroc_per_channel = np.empty(channel_count)

    # Method 3: abs(diff(TS)) > b

    print('Method 3:')
    
    for channel in trange(channel_count,
                            desc='Absdiff: Per-channel '
                                        'AUROC computation'):
        
        auroc_method_3 = roc_auc_score(labels[:, channel],
                                        abs_diff_normalized[:, channel])

        auroc_per_channel[channel] = auroc_method_3

    auroc_min = np.min(auroc_per_channel)
    auroc_max = np.max(auroc_per_channel)
    auroc_mean = np.mean(auroc_per_channel)
    auroc_median = np.median(auroc_per_channel)

    print(f'AUROC: min: {auroc_min:.3f}\t'
                    f'max: {auroc_max:.3f}\t'
                    f'mean: {auroc_mean:.3f}\t'
                    f'median {auroc_median:.3f}')

    mcc_best_per_channel = np.full(channel_count, -1.)
    b_best_per_channel = np.full(channel_count, 0.)

    for b in b_array:

        for channel in range(channel_count):

            data_channel = abs_diff_normalized[:, channel]
            labels_channel = labels[:, channel]

            preds_channel =\
                adjust_predicts(data_channel,
                                    labels_channel, b)
            
            mcc = matthews_corrcoef(labels_channel,
                                        preds_channel)

            if mcc_best_per_channel[channel] < mcc:

                mcc_best_per_channel[channel] = mcc
                b_best_per_channel[channel] = b

    results_best = pd.DataFrame(index=np.arange(mcc_best_per_channel.shape[-1]),
                                                                columns=['mcc', 'b'])

    for channel, (mcc, b) in enumerate(zip(mcc_best_per_channel,
                                                b_best_per_channel)):

        print(f'Best MCC channel {channel}: {mcc:.3f} for {b=:.3f}')

        results_best.iloc[channel, :] = (mcc, b,)

    results_best.to_csv('parameters_best_method_3_hlt_reduced.csv', sep='\t')

    # Method 4: abs(diff(TS)) > movmean(abs(diff(TS)), k) + c*movstd(diff(TS), k) + b

    print('Method 4:')

    usable_processors = len(os.sched_getaffinity(0))
    process_pool = Pool(usable_processors//2)

    method_4_fixed_k_with_args =\
            partial(method_4_fixed_k,
                        abs_diff=abs_diff_normalized,
                        b_array=b_array,
                        c_array=c_array,
                        labels=labels)

    results = process_pool.map(
                    method_4_fixed_k_with_args, k_array)

    process_pool.close()

    ks, mccs_best_per_channel,\
            bs_best_per_channel,\
            cs_best_per_channel = zip(*results)

    mccs_best_per_channel =\
            np.stack(mccs_best_per_channel)

    bs_best_per_channel =\
            np.stack(bs_best_per_channel)

    cs_best_per_channel =\
            np.stack(cs_best_per_channel)

    # Convert results to Pandas MultiIndex DataFrame
    # to store them

    mccs_best_per_channel_pd =\
            pd.DataFrame(mccs_best_per_channel, index=ks)

    bs_best_per_channel_pd =\
            pd.DataFrame(bs_best_per_channel, index=ks)

    cs_best_per_channel_pd =\
            pd.DataFrame(cs_best_per_channel, index=ks)

    data_per_channel_pd =\
        pd.concat((mccs_best_per_channel_pd,
                        bs_best_per_channel_pd,
                        cs_best_per_channel_pd),
                        keys=('mcc', 'b', 'c'))

    data_per_channel_pd.to_hdf('results_hlt_reduced_2018_one_liner_test.h5',
                                    key='results_hlt_reduced_2018_one_liner_test',
                                    mode='w')

    data_per_channel_pd = pd.read_hdf('results_hlt_reduced_2018_one_liner_test.h5',
                                        key='results_hlt_reduced_2018_one_liner_test')

    # Get index of per-column best MCC
    # to determine best combination
    # of b, c, and k for each column

    mccs_best_per_channel = data_per_channel_pd.xs('mcc', level=0)
    bs_best_per_channel = data_per_channel_pd.xs('b', level=0)
    cs_best_per_channel = data_per_channel_pd.xs('c', level=0)

    indices_best = mccs_best_per_channel.idxmax(axis=0)

    results_best = pd.DataFrame(index=np.arange(mccs_best_per_channel.shape[-1]),
                                                    columns=['mcc', 'b', 'c', 'k'])

    for index_col, k in enumerate(indices_best):

        index_row = k - 2

        mcc = mccs_best_per_channel.iloc[index_row, index_col]
        b = bs_best_per_channel.iloc[index_row, index_col]
        c = cs_best_per_channel.iloc[index_row, index_col]

        print(f'Best MCC channel {index_col}: {mcc:.3f} for {b=:.3f}, {c=:.3f}, {k=}')

        results_best.iloc[index_col, :] = (mcc, b, c, k)

    results_best.to_csv('parameters_best_method_4_hlt_reduced.csv', sep='\t')


def test_thresholds_method_3(data: np.array,
                                labels: np.array):

    results_best_method_3 =\
        pd.read_csv('parameters_best_method_3_hlt_reduced.csv', sep='\t')

    diff = np.diff(data, axis=0)

    labels = labels[1:, :]

    abs_diff = np.abs(diff)

    abs_diff_normalized = MinMaxScaler().fit_transform(abs_diff)
    
    bs = results_best_method_3.iloc[:, 2].to_numpy()
    mccs = results_best_method_3.iloc[:, 1].to_numpy()

    print('Method 3:')

    thresholds = np.arange(0.1, 1, 0.025)

    results_combined = pd.DataFrame(index=thresholds,
                                        columns=['AUROC',
                                                    'F1', 'MCC',
                                                    'Precision',
                                                    'Recall'])

    for threshold in thresholds:

        included_indices = np.where(mccs > threshold)[0]

        preds_all = np.zeros_like(abs_diff_normalized)

        bs_included = bs[included_indices]

        for channel, b in zip(included_indices,
                                    bs_included):

            data_channel = abs_diff_normalized[:, channel]
            labels_channel = labels[:, channel]

            preds_all[:, channel] =\
                adjust_predicts(data_channel,
                                    labels_channel, b)

        preds_reduced = np.any(preds_all, axis=1).astype(np.uint8)
        labels_reduced = np.any(labels, axis=1).astype(np.uint8)

        auroc = roc_auc_score(labels_reduced, preds_reduced)
        f1 = f1_score(labels_reduced, preds_reduced)
        mcc = matthews_corrcoef(labels_reduced, preds_reduced)
        precision = precision_score(labels_reduced, preds_reduced)
        recall = recall_score(labels_reduced, preds_reduced)

        print(f'Threshold: {threshold:.3f}: '
                f'AUROC: {auroc:.3f}\tF1: {f1:.3f}\tMCC: {mcc:.3f}\t'
                f'precision: {precision:.3f}\trecall: {recall:.3f}')

        results_combined.loc[threshold, :] = (auroc, f1, mcc, precision, recall)

    results_combined.to_csv('results_method_3_combined_hlt_reduced.csv', sep='\t')


def test_thresholds_method_4(data: np.array,
                                labels: np.array):

    results_best_method_4 =\
        pd.read_csv('parameters_best_method_4_hlt_reduced.csv', sep='\t')

    diff = np.diff(data, axis=0)

    labels = labels[1:, :]

    abs_diff = np.abs(diff)

    abs_diff_normalized = MinMaxScaler().fit_transform(abs_diff)
    
    parameters = results_best_method_4.iloc[:, 2:].to_numpy()
    mccs = results_best_method_4.iloc[:, 1].to_numpy()

    print('Method 4:')

    thresholds = np.arange(0.1, 1, 0.025)

    results_combined = pd.DataFrame(index=thresholds,
                                        columns=['AUROC',
                                                    'F1', 'MCC',
                                                    'Precision',
                                                    'Recall'])

    for threshold in thresholds:

        included_indices = np.where(mccs > threshold)[0]

        preds_all = method_4_combined(abs_diff_normalized[:, included_indices],
                                                    labels[:, included_indices],
                                                    parameters[included_indices, :])

        preds_reduced = np.any(preds_all, axis=1).astype(np.uint8)
        labels_reduced = np.any(labels, axis=1).astype(np.uint8)

        auroc = roc_auc_score(labels_reduced, preds_reduced)
        f1 = f1_score(labels_reduced, preds_reduced)
        mcc = matthews_corrcoef(labels_reduced, preds_reduced)
        precision = precision_score(labels_reduced, preds_reduced)
        recall = recall_score(labels_reduced, preds_reduced)

        print(f'Threshold: {threshold:.3f}: '
                f'AUROC: {auroc:.3f}\tF1: {f1:.3f}\tMCC: {mcc:.3f}\t'
                f'precision: {precision:.3f}\trecall: {recall:.3f}')

        results_combined.loc[threshold, :] = (auroc, f1, mcc, precision, recall)

    results_combined.to_csv('results_method_4_combined_hlt_reduced.csv', sep='\t')


def run_with_best_parameters_method_3(data: np.array,
                                        labels: np.array,
                                        threshold: np.float64):

    results_best_method_3 =\
        pd.read_csv('parameters_best_method_3_hlt_reduced.csv', sep='\t')

    # pylikwid.markerinit()
    # pylikwid.markerthreadinit()

    # pylikwid.markerstartregion("1lm3_0")
    diff = np.diff(data, axis=0)
    # pylikwid.markerstopregion("1lm3_0")

    # nr_events, eventlist, time, count = pylikwid.markergetregion("1lm3_0")

    # for i, e in enumerate(eventlist):
    #     print(i, e)
    # pylikwid.markerclose()

    labels = labels[1:, :]

    # pylikwid.markerinit()
    # pylikwid.markerthreadinit()
    # pylikwid.markerstartregion("1lm3_1")
    abs_diff = np.abs(diff)
    # pylikwid.markerstopregion("1lm3_1")

    # nr_events, eventlist, time, count = pylikwid.markergetregion("1lm3_1")

    # for i, e in enumerate(eventlist):
    #     print(i, e)
    # pylikwid.markerclose()

    abs_diff_normalized = MinMaxScaler().fit_transform(abs_diff)
    
    bs = results_best_method_3.iloc[:, 2].to_numpy()
    mccs = results_best_method_3.iloc[:, 1].to_numpy()

    print('Method 3:')

    included_indices = np.where(mccs > threshold)[0]

    preds_all = np.zeros_like(abs_diff_normalized)

    bs_included = bs[included_indices]

    # pylikwid.markerinit()
    # pylikwid.markerthreadinit()
    # pylikwid.markerstartregion("1lm3_2")

    for channel, b in zip(included_indices,
                                bs_included):

        data_channel = abs_diff_normalized[:, channel]
        labels_channel = labels[:, channel]

        preds_all[:, channel] =\
            adjust_predicts(data_channel,
                                labels_channel, b)

    # pylikwid.markerstopregion("1lm3_2")

    # nr_events, eventlist, time, count = pylikwid.markergetregion("1lm3_2")

    # for i, e in enumerate(eventlist):
    #     print(i, e)
    # pylikwid.markerclose()


    # save_numpy_array(preds_all, '../../evaluation/combined_detection/predictions/method_3.npy')


def run_with_best_parameters_method_4(data: np.array,
                                        labels: np.array,
                                        threshold: np.float64):

    # pylikwid.markerinit()
    # pylikwid.markerthreadinit()

    results_best_method_4 =\
        pd.read_csv('parameters_best_method_4_hlt_reduced.csv', sep='\t')

    # pylikwid.markerstartregion("1lm4_0")
    # diff = np.diff(data, axis=0)
    # pylikwid.markerstopregion("1lm4_0")

    # nr_events, eventlist, time, count = pylikwid.markergetregion("1lm4_0")

    # for i, e in enumerate(eventlist):
    #     print(i, e)
    # pylikwid.markerclose()

    labels = labels[1:, :]

    # pylikwid.markerinit()
    # pylikwid.markerthreadinit()
    # pylikwid.markerstartregion("1lm4_1")
    # abs_diff = np.abs(diff)
    # pylikwid.markerstopregion("1lm4_1")

    # nr_events, eventlist, time, count = pylikwid.markergetregion("1lm4_1")

    for i, e in enumerate(eventlist):
        print(i, e)
    pylikwid.markerclose()

    abs_diff_normalized = MinMaxScaler().fit_transform(abs_diff)
    
    parameters = results_best_method_4.iloc[:, 2:].to_numpy()

    mccs = results_best_method_4.iloc[:, 1].to_numpy()

    preds_all = np.zeros_like(abs_diff_normalized)

    print('Method 4:')

    included_indices = np.where(mccs > threshold)[0]

    # pylikwid.markerinit()
    # pylikwid.markerthreadinit()
    # pylikwid.markerstartregion("1lm4_2")

    preds_all[:, included_indices] =\
                method_4_combined(abs_diff_normalized[:, included_indices],
                                                labels[:, included_indices],
                                                parameters[included_indices, :])
    
    # pylikwid.markerstopregion("1lm4_2")

    # nr_events, eventlist, time, count = pylikwid.markergetregion("1lm4_2")

    # for i, e in enumerate(eventlist):
    #     print(i, e)
    # pylikwid.markerclose()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HLT One-Liner Test')

    parser.add_argument('--data-dir', type=str, default='../../datasets/hlt')
    parser.add_argument('--k-lower', type=int, default=2)
    parser.add_argument('--k-upper', type=int, default=16)
  
    args = parser.parse_args()

    hlt_data_pd = pd.read_hdf(args.data_dir +\
                                    '/reduced_hlt_dcm_test_set_2018_x.h5')

    hlt_data_pd.fillna(0, inplace=True)

    hlt_data_np = hlt_data_pd.to_numpy()

    labels_pd = pd.read_hdf(args.data_dir +\
                            '/reduced_hlt_dcm_test_set_2018_y.h5')

    labels_np = labels_pd.to_numpy()

    labels_np = np.greater_equal(labels_np, 1)

    parameter_exploration(hlt_data_np,
                                labels_np,
                                k_lower=args.k_lower,
                                k_upper=args.k_upper)

    test_thresholds_method_3(hlt_data_np,
                                    labels_np)

    test_thresholds_method_4(hlt_data_np,
                                    labels_np)

    run_with_best_parameters_method_3(hlt_data_np,
                                                labels_np,
                                                0.675)

    run_with_best_parameters_method_4(hlt_data_np,
                                                labels_np,
                                                0.625)
