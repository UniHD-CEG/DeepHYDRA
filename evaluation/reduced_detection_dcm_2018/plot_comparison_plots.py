
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

    label = np.any(label, axis=1).astype(np.uint8)

    preds_dagmm_train = load_numpy_array('predictions/dagmm_train_seed_42.npy')
    preds_usad_train = load_numpy_array('predictions/usad_train_seed_42.npy')
    preds_mscred_train = load_numpy_array('predictions/mscred_train_seed_42.npy')

    preds_dagmm = load_numpy_array('predictions/dagmm_seed_42.npy')
    preds_usad = load_numpy_array('predictions/usad_seed_42.npy')
    preds_mscred = load_numpy_array('predictions/mscred_seed_42.npy')


    # preds_dagmm = np.any(preds_dagmm, axis=1).astype(np.uint8)
    # preds_usad = np.any(preds_usad, axis=1).astype(np.uint8)
    # preds_mscred = np.any(preds_mscred, axis=1).astype(np.uint8)

    spot_train_size = int(len(preds_dagmm)*0.1)

    preds_dagmm =\
        get_thresholded_tranad(preds_dagmm_train[:spot_train_size],
                                            preds_dagmm, label, 0.01)
    
    preds_usad =\
        get_scores_dagmm('usad', 42,
                            preds_usad_train[:spot_train_size],
                            preds_usad,
                            label, 0.0000000001, 0.02, to_csv)

    print('MSCRED:')

    preds_mscred =\
        get_scores_dagmm('mscred', 42,
                            preds_mscred_train[:spot_train_size],
                            preds_mscred,
                            label, 0.0000000001, 0.02, to_csv)
    



    preds_all = {   'DAGMM': preds_dagmm,
                    'USAD': preds_usad,
                    'MSCRED': preds_mscred,}

    # These colors are specifically chosen to improve
    # accessibility for readers with colorblindness

    colors = {  'DAGMM': '#D81B60',
                'USAD': '#1E88E5',
                'MSCRED': '#FFC107',}

    positions = {   'DAGMM': 0,
                    'USAD': 1,
                    'MSCRED': 2,}
    
    SMALL_SIZE = 13
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 13

    xlims = [(600, 800),
                (2000, 2400),
                (6000, 8500),
                (14000, 15000)]
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    x = np.arange(len(data))*5/3600
    
    for index, (xlim_lower, xlim_upper) in enumerate(tqdm(xlims,
                                                    desc='Plotting')):

        fig, (ax_data, ax_pred) = plt.subplots(2, 1, figsize=(10, 6), dpi=300)

        ax_data.set_title('Event Rate')
        ax_data.set_xlabel('Time [h]')
        ax_data.set_ylabel('Event Rate [Hz]')

        ax_data.set_xlim(x[xlim_lower],
                            x[xlim_upper])
        
        ax_data.set_ylim(-1, 100)

        ax_data.grid()

        ax_data.plot(x[xlim_lower:xlim_upper],
                        data[xlim_lower:xlim_upper, :],
                        linewidth=0.9,
                        color='k')

        anomaly_starts, anomaly_ends =\
                    get_anomalous_runs(label)

        for start, end in zip(anomaly_starts,
                                    anomaly_ends):
            ax_data.axvspan(x[start], x[end], color='red', alpha=0.5)
            ax_pred.axvspan(x[start], x[end], color='red', alpha=0.5)

        ax_pred.set_yticks(list(positions.values()),
                                list(positions.keys()))

        ax_pred.set_title('Predictions')
        ax_pred.set_xlabel('Time [h]')
        ax_pred.set_ylabel('Method')

        ax_pred.set_xlim(x[xlim_lower],
                            x[xlim_upper])

        # ax_pred.set_yticks(list(positions.values()),
        #                         list(positions.keys()))
        
        ax_pred.set_yticks(list(positions.values()))
        
        ax_pred.set_yticklabels(list(positions.keys()))
        
        plt.setp(ax_pred.get_yticklabels(),
                                rotation=30,
                                ha="right",
                                rotation_mode="anchor")

        for method, preds in preds_all.items():
            pred_starts, pred_ends =\
                get_anomalous_runs(preds)
            
            print(len(pred_starts))
            print(len(pred_ends))

            exit()
                
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
        plt.savefig(f'plots/prediction_comparison_{index}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HLT Dataset Comparison Plot Generator')

    parser.add_argument('--data-dir', type=str, default='../../datasets/hlt')
  
    args = parser.parse_args()

    hlt_data_pd = pd.read_hdf(args.data_dir +\
                                    '/reduced_hlt_dcm_test_set_2018_x.h5')

    hlt_data_pd.iloc[run_endpoints[-2]:-1, :] = 0

    hlt_data_pd.fillna(0, inplace=True)

    hlt_data_np = hlt_data_pd.to_numpy()

    labels_pd = pd.read_hdf(args.data_dir +\
                            '/reduced_hlt_dcm_test_set_2018_y.h5')

    labels_np = labels_pd.to_numpy()

    labels_np = np.greater_equal(labels_np, 1)

    plot_results(hlt_data_np,
                    labels_np)
