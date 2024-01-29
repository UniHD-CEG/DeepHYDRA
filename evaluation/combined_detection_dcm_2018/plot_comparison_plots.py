
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

mpl.rcParams['hatch.linewidth'] = 9.0

sys.path.append('.')

from spot import SPOT

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


def get_thresholded_dagmm(pred_train,
                            pred_test,
                            true,
                            q=1e-3,
                            level=0.02,
                            thresh_tweak_factor=20):
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
    pot_th = np.mean(ret['thresholds'])*thresh_tweak_factor
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    pred = pred_test > pot_th

    pred = adjust_predicts(pred, true, 0.1)

    return pred


def plot_results(data: np.array,
                    label: np.array):

    label = np.any(label, axis=1).astype(np.uint8)

    preds_method_3 = load_numpy_array('predictions/method_3.npy')
    preds_method_4 = load_numpy_array('predictions/method_4.npy')
    preds_merlin = load_numpy_array('predictions/merlin.npy')
    preds_clustering = load_numpy_array('predictions/clustering.npy')
    preds_tranad = load_numpy_array('predictions/tranad_seed_7.npy')
    preds_tranad_train = load_numpy_array(f'predictions/tranad_train_seed_7.npy')
    preds_l2_dist_train_mse = load_numpy_array(f'predictions/l2_dist_train_mse_seed_192.npy')
    preds_l2_dist_mse = load_numpy_array(f'predictions/l2_dist_mse_seed_192.npy')
    preds_l2_dist_train_smse = load_numpy_array(f'predictions/l2_dist_train_smse_seed_85.npy')
    preds_l2_dist_smse = load_numpy_array(f'predictions/l2_dist_smse_seed_85.npy')
    preds_omni_anomaly = load_numpy_array('predictions/omnianomaly_seed_28.npy')
    preds_omni_anomaly_train = load_numpy_array(f'predictions/omnianomaly_train_seed_28.npy')
    preds_dagmm = load_numpy_array('predictions/dagmm_seed_142.npy')
    preds_dagmm_train = load_numpy_array(f'predictions/dagmm_train_seed_142.npy')
    preds_usad = load_numpy_array('predictions/usad_seed_129.npy')
    preds_usad_train = load_numpy_array(f'predictions/usad_train_seed_129.npy')

    preds_method_3 = np.any(preds_method_3, axis=1).astype(np.uint8)
    preds_method_4 = np.any(preds_method_4, axis=1).astype(np.uint8)
    preds_merlin = np.any(preds_merlin, axis=1).astype(np.uint8)

    spot_train_size = int(len(preds_l2_dist_mse)*0.1)

    # Fix alignment

    # pbar = tqdm(total=7, desc='Preprocessing')

    preds_l2_dist_mse =\
        np.pad(preds_l2_dist_mse[1:], (0, 1),
                                'constant',
                                constant_values=(0,))
    
    preds_l2_dist_smse =\
        np.pad(preds_l2_dist_smse[1:], (0, 1),
                                'constant',
                                constant_values=(0,))
    
    preds_method_3 =\
        np.pad(preds_method_3, (1, 0),
                            'constant',
                            constant_values=(0,))
    preds_method_4 =\
        np.pad(preds_method_4, (1, 0),
                            'constant',
                            constant_values=(0,))
    
    preds_clustering =\
        adjust_predicts(preds_clustering,
                                label, 0.1)
    
    # pbar.update(1)

    # preds_tranad =\
    #     get_thresholded_tranad(preds_tranad_train,
    #                         preds_tranad, label, 0.01)

    

    # preds_deep_hydra_tranad = np.logical_or(preds_clustering,
    #                                             preds_tranad)

    # pbar.update(1)

    # preds_l2_dist_mse =\
    #     get_thresholded(preds_l2_dist_train_mse[:spot_train_size],
    #                                             preds_l2_dist_mse, label, 0.0025)

    # preds_deep_hydra_mse = np.logical_or(preds_clustering,
    #                                     preds_l2_dist_mse)

    # pbar.update(1)

    # preds_l2_dist_smse =\
    #     get_thresholded(preds_l2_dist_train_smse[:spot_train_size],
    #                                             preds_l2_dist_smse, label, 0.008)
    
    # preds_deep_hydra_smse = np.logical_or(preds_clustering,
    #                                     preds_l2_dist_smse)

    # pbar.update(1)

    # preds_dagmm = get_thresholded_dagmm(preds_dagmm_train[:spot_train_size],
    #                             preds_dagmm,
    #                             label, 0.008,
    #                             0.8, 60)
    
    # preds_deep_hydra_dagmm =\
    #     np.logical_or(preds_clustering,
    #                         preds_dagmm)

    # pbar.update(1)

    # preds_omni_anomaly = preds_omni_anomaly[4:]

    # preds_omni_anomaly = get_thresholded_dagmm(preds_omni_anomaly_train[:spot_train_size],
    #                                     preds_omni_anomaly,
    #                                     label, 0.0001, 0.8, 1.3)

    # preds_deep_hydra_omni_anomaly =\
    #     np.logical_or(preds_clustering,
    #                         preds_omni_anomaly)

    # pbar.update(1)

    # preds_usad = get_thresholded_dagmm(preds_usad_train[:spot_train_size],
    #                                         preds_usad,
    #                                         label, 0.01, 0.02, 18,)

    # preds_deep_hydra_usad =\
    #     np.logical_or(preds_clustering,
    #                             preds_usad)

    # pbar.close()

    # save_numpy_array(preds_tranad, 'predictions_thresholded/preds_tranad.npy')
    # save_numpy_array(preds_deep_hydra_tranad, 'predictions_thresholded/preds_deep_hydra_tranad.npy')
    # save_numpy_array(preds_l2_dist_mse, 'predictions_thresholded/preds_l2_dist_mse.npy')
    # save_numpy_array(preds_deep_hydra_mse, 'predictions_thresholded/preds_deep_hydra_mse.npy')
    # save_numpy_array(preds_l2_dist_smse, 'predictions_thresholded/preds_l2_dist_smse.npy')
    # save_numpy_array(preds_deep_hydra_smse, 'predictions_thresholded/preds_deep_hydra_smse.npy')
    # save_numpy_array(preds_omni_anomaly, 'predictions_thresholded/preds_omni_anomaly.npy')
    # save_numpy_array(preds_deep_hydra_omni_anomaly, 'predictions_thresholded/preds_deep_hydra_omni_anomaly.npy')
    # save_numpy_array(preds_dagmm, 'predictions_thresholded/preds_dagmm.npy')
    # save_numpy_array(preds_deep_hydra_dagmm, 'predictions_thresholded/preds_deep_hydra_dagmm.npy')
    # save_numpy_array(preds_usad, 'predictions_thresholded/preds_usad.npy')
    # save_numpy_array(preds_deep_hydra_usad, 'predictions_thresholded/preds_deep_hydra_usad.npy')

    # exit()

    preds_tranad = load_numpy_array('predictions_thresholded/preds_tranad.npy')
    preds_deep_hydra_tranad = load_numpy_array(f'predictions_thresholded/preds_deep_hydra_tranad.npy')
    preds_l2_dist_mse = load_numpy_array(f'predictions_thresholded/preds_l2_dist_mse.npy')
    preds_deep_hydra_mse = load_numpy_array(f'predictions_thresholded/preds_deep_hydra_mse.npy')
    preds_l2_dist_smse = load_numpy_array(f'predictions_thresholded/preds_l2_dist_smse.npy')
    preds_deep_hydra_smse = load_numpy_array(f'predictions_thresholded/preds_deep_hydra_smse.npy')
    preds_omni_anomaly = load_numpy_array('predictions_thresholded/preds_omni_anomaly.npy')
    preds_deep_hydra_omni_anomaly = load_numpy_array(f'predictions_thresholded/preds_deep_hydra_omni_anomaly.npy')
    preds_dagmm = load_numpy_array('predictions_thresholded/preds_dagmm.npy')
    preds_deep_hydra_dagmm = load_numpy_array(f'predictions_thresholded/preds_deep_hydra_dagmm.npy')
    preds_usad = load_numpy_array('predictions_thresholded/preds_usad.npy')
    preds_deep_hydra_usad = load_numpy_array(f'predictions_thresholded/preds_deep_hydra_usad.npy')

    preds_all = {   '1L-Method 3': preds_method_3,
                    '1L-Method 4': preds_method_4,
                    'MERLIN': preds_merlin,
                    'T-DBSCAN': preds_clustering,
                    'OmniAnomaly': preds_omni_anomaly,
                    'DeepHYDRA-OmniAnomaly': preds_deep_hydra_omni_anomaly,
                    'DAGMM': preds_dagmm,
                    'DeepHYDRA-DAGMM': preds_deep_hydra_dagmm,
                    'USAD': preds_usad,
                    'DeepHYDRA-USAD': preds_deep_hydra_usad,
                    'Informer-MSE': preds_l2_dist_mse,
                    'DeepHYDRA-MSE': preds_deep_hydra_mse,
                    'Informer-SMSE': preds_l2_dist_smse,
                    'DeepHYDRA-SMSE': preds_deep_hydra_smse,
                    'TranAD': preds_tranad,
                    'DeepHYDRA-TranAD': preds_deep_hydra_tranad}

    # These colors are specifically chosen to improve
    # accessibility for readers with colorblindness

    colors = {  '1L-Method 3': '#D81B60',
                '1L-Method 4': '#1E88E5',
                'MERLIN': '#FFC107',
                'T-DBSCAN': '#000000', # '#004D40'
                'OmniAnomaly': '#D4FC14',
                'DeepHYDRA-OmniAnomaly': '#D4FC14',
                'DAGMM': '#6F8098',
                'DeepHYDRA-DAGMM': '#6F8098',
                'USAD': '#663399',
                'DeepHYDRA-USAD': '#663399',
                'Informer-MSE': '#1CB2C5',
                'DeepHYDRA-MSE': '#1CB2C5',
                'Informer-SMSE': '#18F964',
                'DeepHYDRA-SMSE': '#18F964',
                'TranAD': '#1164B3',
                'DeepHYDRA-TranAD': '#1164B3'}

    positions = {   '1L-Method 3': 15,
                    '1L-Method 4': 14,
                    'MERLIN': 13,
                    'T-DBSCAN': 12,
                    'OmniAnomaly': 11,
                    'DeepHYDRA-OmniAnomaly': 10,
                    'DAGMM': 9,
                    'DeepHYDRA-DAGMM': 8,
                    'USAD': 7,
                    'DeepHYDRA-USAD':6,
                    'Informer-MSE': 5,
                    'DeepHYDRA-MSE': 4,
                    'Informer-SMSE': 3,
                    'DeepHYDRA-SMSE': 2,
                    'TranAD': 1,
                    'DeepHYDRA-TranAD': 0}
    
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

        fig, (ax_data, ax_pred) = plt.subplots(2, 1, figsize=(8, 7),
                                    gridspec_kw={'height_ratios': [1, 2]}, dpi=300)

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
                
            for start, end in zip(pred_starts, pred_ends):

                length = x[end] - x[start]

                if 'DeepHYDRA' in method:
                    ax_pred.barh(positions[method],
                                    length,
                                    left=x[start],
                                    color=colors[method],
                                    edgecolor='k',
                                    hatch='/',
                                    linewidth=0.7,
                                    label=method,
                                    height=0.85)
                    
                else:
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
                                    '/unreduced_hlt_dcm_test_set_2018_x.h5')

    hlt_data_pd.iloc[run_endpoints[-2]:-1,
                            channels_to_delete_last_run] = 0

    hlt_data_pd.fillna(0, inplace=True)

    hlt_data_np = hlt_data_pd.to_numpy()

    labels_pd = pd.read_hdf(args.data_dir +\
                            '/unreduced_hlt_dcm_test_set_2018_y.h5')

    labels_np = labels_pd.to_numpy()

    labels_np = np.greater_equal(labels_np, 1)

    plot_results(hlt_data_np,
                    labels_np)
