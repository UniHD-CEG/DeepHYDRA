
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('.')

from spot import SPOT

run_endpoints = [1404,
                    8928,
                    19296,
                    28948]

channels_to_delete_last_run = [1357,
                                3685,
                                3184]


def load_numpy_array(filename: str):
    with open(filename, 'rb') as output_file:
        return np.load(output_file)


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


def get_thresholded(pred_train, pred_test, true, q=1e-3, level=0.8):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """

    # SPOT object
    spot = SPOT(q)

    # data import
    spot.fit(pred_train, pred_test)

    # initialization step
    spot.initialize(level=level,
                    min_extrema=False,
                    verbose=False)

    # run
    ret = spot.run()

    pred = np.zeros_like(pred_test, dtype=np.uint8)

    pred[ret['alarms']] = 1

    pred = adjust_predicts(pred, true, 0.1)

    return pred


def get_thresholded_tranad(pred_train, pred_test, true, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """

    lms = 0.99995
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(pred_train, pred_test)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * 0.6
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    pred = pred_test > pot_th

    pred = adjust_predicts(pred, true, 0.1)

    return pred


def plot_results(data: np.array,
                    label: np.array):
    
    SMALL_SIZE = 13
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 13

    x = np.arange(len(data))*5/3600

    seeds = [7, 129, 28, 192, 85, 148, 142, 30, 78, 33]



    for seed in tqdm(seeds, desc='Plotting'):

        preds_l2_dist_train_mse =\
            load_numpy_array(f'predictions/l2_dist_train_mse_seed_{seed}.npy')
        preds_l2_dist_mse =\
            load_numpy_array(f'predictions/l2_dist_mse_seed_{seed}.npy')
        preds_l2_dist_train_smse =\
            load_numpy_array(f'predictions/l2_dist_train_smse_seed_{seed}.npy')
        preds_l2_dist_smse =\
            load_numpy_array(f'predictions/l2_dist_smse_seed_{seed}.npy')

        spot_train_size = int(len(preds_l2_dist_mse)*0.1)

        print('Informer-MSE:')

        offset = 16

        preds_l2_dist_mse =\
            get_thresholded(preds_l2_dist_train_mse[:spot_train_size],
                                                    preds_l2_dist_mse,
                                                    label[offset:len(preds_l2_dist_mse) + offset],
                                                    0.0008, 0.8)

        print('Informer-SMSE:')

        offset = 64

        preds_l2_dist_smse =\
            get_thresholded(preds_l2_dist_train_smse[:spot_train_size],
                                                    preds_l2_dist_smse,
                                                    label[offset:len(preds_l2_dist_smse) + offset],
                                                    0.0008, 0.8)

        preds_all = {   'Informer-MSE': preds_l2_dist_mse,
                        'Informer-SMSE': preds_l2_dist_smse,}

        # These colors are specifically chosen to improve
        # accessibility for readers with colorblindness

        colors = {  'Informer-MSE': '#D81B60',
                    'Informer-SMSE': '#1E88E5',}

        positions = {   'Informer-MSE': 0,
                        'Informer-SMSE': 1,}

        fig, (ax_data, ax_pred) = plt.subplots(2, 1, figsize=(10, 6), dpi=300)

        ax_data.set_title('Data')
        ax_data.set_xlabel('Time [h]')
        ax_data.set_ylabel('Data')

        ax_data.grid()

        ax_data.plot(x, data,
                        linewidth=0.9,
                        color='k')

        anomaly_starts, anomaly_ends =\
                    get_anomalous_runs(label)

        for start, end in zip(anomaly_starts,
                                    anomaly_ends):

            end = end if end < (len(data) - 1) else (len(data) - 1)

            ax_data.axvspan(x[start], x[end], color='red', alpha=0.5)
            ax_pred.axvspan(x[start], x[end], color='red', alpha=0.5)

        ax_pred.set_yticks(list(positions.values()),
                                list(positions.keys()))

        ax_pred.set_title('Predictions')
        ax_pred.set_xlabel('Time [h]')
        ax_pred.set_ylabel('Method')
        
        ax_pred.set_yticks(list(positions.values()))
        ax_pred.set_yticklabels(list(positions.keys()))
        
        plt.setp(ax_pred.get_yticklabels(),
                                rotation=30,
                                ha="right",
                                rotation_mode="anchor")

        for method, preds in preds_all.items():
            pred_starts, pred_ends =\
                get_anomalous_runs(preds)
                
            for start, end in zip(pred_starts, pred_ends):

                end = min(end, (len(x) - 1))

                length = x[end] - x[start]

                ax_pred.barh(positions[method],
                                length,
                                left=x[start],
                                color=colors[method],
                                edgecolor='k',
                                linewidth=0.7,
                                label=method,
                                height=0.85)

        plt.tight_layout()
        plt.savefig(f'plots/prediction_comparison_seed_{seed}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reduced ECLIPSE Dataset Plotting')

    parser.add_argument('--data-dir', type=str, default='../../datasets/eclipse')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--to-csv', action='store_true', default=False)
  
    args = parser.parse_args()

    data = pd.read_hdf(args.data_dir +\
                            '/reduced_eclipse_labeled_val_set_median.h5')
    
    labels_pd = data.loc[:, data.columns == 'label']

    labels_np = labels_pd.to_numpy()
    labels_np = (labels_np>=1).astype(np.uint8).flatten()

    data = data.loc[:, ~(data.columns.str.contains('label'))]

    data_np = data.to_numpy()

    plot_results(data_np,
                    labels_np)

