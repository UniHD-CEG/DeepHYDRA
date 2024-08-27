#!/usr/bin/env python3

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,\
                                    f1_score,\
                                    precision_score,\
                                    recall_score,\
                                    matthews_corrcoef
from tqdm import trange

from merlin import merlin
import pylikwid


def save_numpy_array(array: np.array,
                        filename: str):    
    with open(filename, 'wb') as output_file:
        np.save(output_file, array)


def load_numpy_array(filename: str):
    with open(filename, 'rb') as output_file:
        return np.load(output_file)


def preds_from_discords(discords: np.array,
                            lengths: np.array,
                            labels: np.array,
                            data_len: int,
                            tolerance_period: int = 100) -> np.array:

    columns = discords.shape[-1]

    preds = np.zeros((data_len, columns))

    for column in trange(columns,
                            desc='Generating preds '
                                    'from discords'):

        anomaly_starts, anomaly_ends =\
            get_anomalous_runs(labels[:, column])

        for discord, length in zip(discords[:, column],
                                        lengths[:, column]):
            pred_start = discord
            pred_end = discord + length + 1
            
            for actual_start, actual_end in\
                    zip(anomaly_starts, anomaly_ends):

                actual_start_with_tol = max(0, actual_start -\
                                                tolerance_period)

                if max(actual_start_with_tol, pred_start) <\
                                        min(actual_end, pred_end):

                    assert actual_start != actual_end

                    preds[actual_start:actual_end, column] = 1
                    
                else:
                    preds[discord, column] = 1

    return preds


def get_anomalous_runs(x):
    '''
    Find runs of consecutive items in an array.
    As published in https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    '''

    # Ensure array

    x = np.asanyarray(x)

    if x.ndim != 1:
        raise ValueError('Only 1D arrays supported')

    n = x.shape[0]

    # Handle empty array

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:

        # Find run starts

        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True

        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # Find run values
        run_values = x[loc_run_start]

        # Find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        run_starts = np.compress(run_values, run_starts)
        run_lengths = np.compress(run_values, run_lengths)

        run_ends = run_starts + run_lengths

        return run_starts, run_ends


def expand_labels(data_columns: pd.Index,
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


def adjust_predicts(score: np.ndarray,
                    label: np.ndarray,
                    threshold=None,
                    pred=None,
                    calc_latency: bool=False):
    '''
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
    '''

    if len(score) != len(label):
        raise ValueError('score and label must have the same length')

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
        return predict, latency/(anomaly_count + 1e-4)
    else:
        return predict


def run_merlin(data: np.ndarray,
                        label: np.ndarray,
                        l_min: int,
                        l_max: int,
                        near_constant_fix: bool,
                        variant: str):

    columns = data.shape[-1]

    # discords_all = []
    # distances_all = []
    # lengths_all = []

    # parameters_all = []

    # flops_all = []

    # for channel in trange(columns):

    #     # pylikwid.markerinit()
    #     # pylikwid.markerthreadinit()
    #     # pylikwid.markerstartregion("MERLIN")

    #     discords, distances, lengths, parameters =\
    #                             merlin(data[:, channel],
    #                                         l_min, l_max,
    #                                         sanitize=near_constant_fix)
        
    #     pylikwid.markerstopregion("MERLIN")

    #     nr_events, eventlist, time, count = pylikwid.markergetregion("MERLIN")

    #     flops_all.append(eventlist[4])
        
    #     pylikwid.markerclose()

    #     parameters_all.append(parameters)

    # print(f'Size MERLIN sequential: {np.max(parameters_all)}')
    # print(f'Size MERLIN parallel: {np.sum(parameters_all)}')

    # print(flops_all)
    
    # exit()

    # for channel in trange(columns):
    #     discords, distances, lengths, parameters =\
    #                             merlin(data[:, channel],
    #                                         l_min, l_max,
    #                                         sanitize=near_constant_fix)

    #     parameters_all.append(parameters)

    #     discords_all.append(discords)
    #     distances_all.append(distances)
    #     lengths_all.append(lengths)

    # print(f'Size MERLIN sequential: {np.max(parameters_all)}')
    # print(f'Size MERLIN parallel: {np.sum(parameters_all)}')
    
    # discords_all = np.column_stack(discords_all)
    # distances_all = np.column_stack(distances_all)
    # lengths_all = np.column_stack(lengths_all)

    # save_numpy_array(distances_all, f'distances_eclipse_reduced_{variant}.npy')
    # save_numpy_array(discords_all, f'discords_eclipse_reduced_{variant}.npy')
    # save_numpy_array(lengths_all, f'lengths_eclipse_reduced_{variant}.npy')

    distances_all = load_numpy_array(f'distances_eclipse_reduced_{variant}.npy')
    discords_all = load_numpy_array(f'discords_eclipse_reduced_{variant}.npy')
    lengths_all = load_numpy_array(f'lengths_eclipse_reduced_{variant}.npy')

    pred = preds_from_discords(discords_all,
                                    lengths_all,
                                    label,
                                    len(data))

    mccs = np.empty((columns,))

    for column in range(columns):

        pred_column = pred[:, column]
        label_column = label[:, column]

        auroc = roc_auc_score(label_column, pred_column)
        f1 = f1_score(label_column, pred_column)
        mcc = matthews_corrcoef(label_column, pred_column)
        precision = precision_score(label_column, pred_column)
        recall = recall_score(label_column, pred_column)

        mccs[column] = mcc

        print(f'Channel {column}: '
                f'AUROC: {auroc:.3f}'
                f'F1: {f1:.3f}'
                f'\tMCC: {mcc:.3f}'
                f'\tPrecision: {precision:.3f}'
                f'\tRecall: {recall:.3f}')

    thresholds = np.arange(0.1, 1, 0.025)

    results_combined = pd.DataFrame(index=thresholds,
                                        columns=['AUROC',
                                                    'F1', 'MCC',
                                                    'Precision',
                                                    'Recall'])

    for threshold in thresholds:

        included_indices = np.where(mccs > threshold)[0]

        pred_reduced = pred[:, included_indices]

        pred_reduced = np.any(pred_reduced, axis=1).astype(np.uint8)
        label_reduced = np.any(label, axis=1).astype(np.uint8)

        auroc = roc_auc_score(label_reduced, pred_reduced)
        f1 = f1_score(label_reduced, pred_reduced)
        mcc = matthews_corrcoef(label_reduced, pred_reduced)
        precision = precision_score(label_reduced, pred_reduced)
        recall = recall_score(label_reduced, pred_reduced)

        print(f'Threshold: {threshold:.3f}: '
                        f'AUROC: {auroc:.3f}\t'
                        f'F1: {f1:.3f}\tMCC: {mcc:.3f}\t'\
                        f'precision: {precision:.3f}'
                        f'\trecall: {recall:.3f}')

        results_combined.loc[threshold, :] = (auroc, f1, mcc, precision, recall)

    results_combined.to_csv(f'results_merlin_reduced_eclipse_{variant}.csv', sep='\t')


def get_preds_best_threshold(data: np.ndarray,
                                label: np.ndarray,
                                threshold: np.float64,
                                variant: str):

    columns = data.shape[-1]

    distances_all = load_numpy_array(f'distances_eclipse_reduced_{variant}.npy')
    discords_all = load_numpy_array(f'discords_eclipse_reduced_{variant}.npy')
    lengths_all = load_numpy_array(f'lengths_eclipse_reduced_{variant}.npy')

    pred = preds_from_discords(discords_all,
                                    lengths_all,
                                    label,
                                    len(data))

    mccs = np.empty((columns,))

    for column in range(columns):

        pred_column = pred[:, column]
        label_column = label[:, column]

        auroc = roc_auc_score(label_column, pred_column)
        f1 = f1_score(label_column, pred_column)
        mcc = matthews_corrcoef(label_column, pred_column)
        precision = precision_score(label_column, pred_column)
        recall = recall_score(label_column, pred_column)

        mccs[column] = mcc

        print(f'Channel {column}: '
                f'AUROC: {auroc:.3f}'
                f'F1: {f1:.3f}'
                f'\tMCC: {mcc:.3f}'
                f'\tPrecision: {precision:.3f}'
                f'\tRecall: {recall:.3f}')


    included_indices = np.where(mccs > threshold)[0]

    pred_reduced = np.zeros_like(pred)
    pred_reduced[:, included_indices] =\
                    pred[:, included_indices]
    
    save_numpy_array(pred_reduced, f'../../evaluation/reduced_detection_eclipse_{variant}/predictions/merlin.npy')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MERLIN ECLIPSE Test')

    parser.add_argument('--data-dir', type=str, default='../../datasets/eclipse')
    parser.add_argument('--variant', type=str, default='median')
    parser.add_argument('--l-min', type=int, default=8)
    parser.add_argument('--l-max', type=int, default=96)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--no-near-constant-fix', action='store_true', default=False)
    parser.add_argument('--log-level', type=str, default='INFO')
  
    args = parser.parse_args()

    logging_format = '[%(asctime)s] %(message)s'

    logger = logging.getLogger(__name__)

    logging.basicConfig(level=args.log_level.upper(),
                                    format=logging_format,
                                    datefmt='%Y-%m-%d %H:%M:%S')

    eclipse_data_pd = pd.read_hdf(args.data_dir +\
                                    '/reduced_eclipse_'
                                    'test_set_'
                                    f'{args.variant}.h5',
                                    key='data')

    eclipse_data_np = eclipse_data_pd.to_numpy()
        
    labels_pd = pd.read_hdf(args.data_dir +\
                                '/reduced_eclipse_'
                                'test_set_'
                                f'{args.variant}.h5',
                                key='labels')

    labels_pd = expand_labels(eclipse_data_pd.columns,
                                                labels_pd)

    labels_np = labels_pd.to_numpy()

    labels_np = np.greater_equal(labels_np, 1)

    run_merlin(eclipse_data_np,
                        labels_np,
                        args.l_min,
                        args.l_max,
                        not args.no_near_constant_fix,
                        args.variant)

    get_preds_best_threshold(eclipse_data_np,
                                    labels_np,
                                    0.9,
                                    args.variant)