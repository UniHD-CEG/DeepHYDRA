

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,\
                                precision_recall_fscore_support,\
                                matthews_corrcoef

sys.path.append('.')

from spot import SPOT

run_endpoints = [1404,
                    8928,
                    19296,
                    28948]

channels_to_delete_last_run = [1357,
                                3685,
                                3184]

base_data_anomaly_starts = [247,
                                465,
                                4272]

base_data_anomaly_ends = [264,
                            465,
                            4277]

anomaly_categories = {'Point Global': 0b0000001,
                        'Point Contextual': 0b0000010,
                        'Persistent Global': 0b0000100,
                        'Persistent Contextual': 0b0001000,
                        'Collective Global': 0b0010000,
                        'Collective Trend': 0b0100000,
                        'Intra Rack': 0b1000000}


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


def metric_comparison_by_categories(pred,
                                        true):

    true_reduced = np.bitwise_or.reduce(true, axis=1)

    preds_adjusted_per_category = []

    print('By Category:')

    for category, flag in anomaly_categories.items():
        print(category)

        mask = np.where(true_reduced & flag, 1, 0)
        mask = np.logical_or(mask,
                    np.where(true_reduced == 0, 1, 0))
        
        true_masked = np.ma.masked_where(~mask, true_reduced)
        pred_masked = np.ma.masked_where(~mask, pred)

        true_masked = np.ma.compressed(true_masked)
        pred_masked = np.ma.compressed(pred_masked)

        true_masked = np.greater_equal(true_masked, 1).astype(np.uint8)

        pred_masked = adjust_predicts(pred_masked,
                                        true_masked, 0.1)
        
        preds_adjusted_per_category.append(pred_masked)
        
        auroc = roc_auc_score(true_masked, pred_masked)

        precision,\
            recall,\
            f1, _ = precision_recall_fscore_support(true_masked,
                                                        pred_masked,
                                                        average='binary')

        mcc = matthews_corrcoef(true_masked,
                                    pred_masked)

        print(f'AUROC: {auroc:.3f}'
                f'\tF1: {f1:.3f}'
                f'\tMCC: {mcc:.3f}'
                f'\tPrecision: {precision:.3f}'
                f'\tRecall: {recall:.3f}')

    return preds_adjusted_per_category


def get_scores(pred_train, pred_test, true, q=1e-3, level=0.8):
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

    true_reduced =\
        np.any(np.greater_equal(true, 1), axis=1).astype(np.uint8)

    # SPOT object
    spot = SPOT(q)

    # data import
    spot.fit(pred_train, pred_test)

    # initialization step
    spot.initialize(level=level,
                    min_extrema=False,
                    verbose=True)

    # run
    ret = spot.run()

    pred = np.zeros_like(pred_test, dtype=np.uint8)

    pred[ret['alarms']] = 1

    preds_adjusted_by_category =\
        metric_comparison_by_categories(pred, true)

    pred = adjust_predicts(pred, true_reduced, 0.1)

    precision_max_f1_score,\
        recall_max_f1_score,\
        max_f1_score, _ = precision_recall_fscore_support(true_reduced,
                                                                    pred,
                                                                    average='binary')

    mcc_max_f1_score =\
        matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {max_f1_score:.3f}\t'
            f'MCC: {mcc_max_f1_score:.3f}\t'
            f'Precision: {precision_max_f1_score:.3f}\t'
            f'Recall: {recall_max_f1_score:.3f}')

    return pred, preds_adjusted_by_category


def get_scores_tranad(pred_train, pred_test, true, q=1e-3, level=0.02):
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

    true_reduced =\
        np.any(np.greater_equal(true, 1), axis=1).astype(np.uint8)

    lms = 0.99995
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(pred_train, pred_test)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=True)  # initialization step
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * 0.3
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    pred = pred_test > pot_th

    preds_adjusted_by_category =\
        metric_comparison_by_categories(pred, true)

    pred = adjust_predicts(pred, true_reduced, 0.1)

    precision_max_f1_score,\
        recall_max_f1_score,\
        max_f1_score, _ = precision_recall_fscore_support(true_reduced,
                                                            pred,
                                                            average='binary')

    mcc_max_f1_score =\
        matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {max_f1_score:.3f}\t'
            f'MCC: {mcc_max_f1_score:.3f}\t'
            f'Precision: {precision_max_f1_score:.3f}\t'
            f'Recall: {recall_max_f1_score:.3f}')

    return pred, preds_adjusted_by_category


def get_scores_thresholded(pred, true):

    true_reduced =\
        np.any(np.greater_equal(true, 1), axis=1).astype(np.uint8)

    precision,\
        recall,\
        f1_score, _ = precision_recall_fscore_support(true_reduced,
                                                                pred,
                                                                average='binary')

    mcc =\
        matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {f1_score:.3f}\t'
            f'MCC: {mcc:.3f}\t'
            f'Precision: {precision:.3f}\t'
            f'Recall: {recall:.3f}')


def get_scores_thresholded_by_category(pred_tdbscan,
                                        preds_transformer,
                                        true):

    true_reduced = np.bitwise_or.reduce(true, axis=1)

    print('By Category:')

    for index, (category, flag) in\
            enumerate(anomaly_categories.items()):
        
        pred = preds_transformer[index]

        print(category)

        mask = np.where(true_reduced & flag, 1, 0)
        mask = np.logical_or(mask,
                    np.where(true_reduced == 0, 1, 0))
        
        true_masked = np.ma.masked_where(~mask, true_reduced)
        pred_tdbscan_masked =\
            np.ma.masked_where(~mask, pred_tdbscan)

        true_masked = np.ma.compressed(true_masked)
        pred_tdbscan_masked = np.ma.compressed(pred_tdbscan_masked)

        true_masked = np.greater_equal(true_masked, 1).astype(np.uint8)

        pred_combined =\
            np.logical_or(pred_tdbscan_masked,
                                            pred)

        auroc = roc_auc_score(true_masked, pred_combined)

        precision,\
            recall,\
            f1, _ = precision_recall_fscore_support(true_masked,
                                                        pred_combined,
                                                        average='binary')

        mcc = matthews_corrcoef(true_masked,
                                    pred_combined)

        print(f'AUROC: {auroc:.3f}'
                f'\tF1: {f1:.3f}'
                f'\tMCC: {mcc:.3f}'
                f'\tPrecision: {precision:.3f}'
                f'\tRecall: {recall:.3f}')


def print_results(label: np.array,
                        seed: int):

    preds_clustering =\
        load_numpy_array('predictions/clustering.npy')
    preds_tranad =\
        load_numpy_array(f'predictions/tranad_seed_{seed}.npy')
    preds_tranad_train =\
        load_numpy_array(f'predictions/tranad_train_seed_{seed}.npy')
    preds_l2_dist_train_mse =\
        load_numpy_array(f'predictions/l2_dist_train_mse_seed_{seed}.npy')
    preds_l2_dist_mse =\
        load_numpy_array(f'predictions/l2_dist_mse_seed_{seed}.npy')
    preds_l2_dist_train_smse =\
        load_numpy_array(f'predictions/l2_dist_train_smse_seed_{seed}.npy')
    preds_l2_dist_smse =\
        load_numpy_array(f'predictions/l2_dist_smse_seed_{seed}.npy')

    spot_train_size = int(len(preds_l2_dist_mse)*0.1)

    # Fix alignment
    
    preds_l2_dist_mse =\
        np.pad(preds_l2_dist_mse[1:],
                    (0, 1), 'constant',
                    constant_values=(0,))
    
    preds_l2_dist_smse =\
        np.pad(preds_l2_dist_smse[1:],
                        (0, 1), 'constant',
                        constant_values=(0,))
    
    
    label_reduced =\
        np.any(np.greater_equal(label, 1), axis=1).astype(np.uint8)

    preds_clustering =\
        adjust_predicts(preds_clustering,
                            label_reduced, 0.1)

    print('T-DBSCAN:')

    get_scores_thresholded(preds_clustering,
                                        label)
    
    # metric_comparison_by_categories(preds_clustering,
    #                                             label)

    print('TranAD:')

    preds_tranad,\
        preds_tranad_by_category =\
            get_scores_tranad(preds_tranad_train,
                                    preds_tranad,
                                    label, 0.01)
    
    print('STRADA-TranAD:')

    preds_strada_tranad =\
        np.logical_or(preds_clustering,
                            preds_tranad)
    
    get_scores_thresholded(preds_strada_tranad,
                                            label)

    # get_scores_thresholded_by_category(preds_clustering,
    #                                     preds_tranad_by_category,
    #                                     label)
    
    print('Informer-MSE:')

    preds_l2_dist_mse,\
        preds_l2_dist_mse_by_category =\
            get_scores(preds_l2_dist_train_mse[:spot_train_size],
                                                preds_l2_dist_mse,
                                                label, 0.0025)

    print('STRADA-MSE:')

    preds_strada_mse =\
        np.logical_or(preds_clustering,
                        preds_l2_dist_mse)
    
    get_scores_thresholded(preds_strada_mse,
                                        label)
    
    # get_scores_thresholded_by_category(preds_clustering,
    #                                     preds_l2_dist_mse_by_category,
    #                                     label)

    print('Informer-SMSE:')

    preds_l2_dist_smse,\
        preds_l2_dist_smse_by_category =\
            get_scores(preds_l2_dist_train_smse[:spot_train_size],
                                                preds_l2_dist_smse,
                                                label, 0.008)
    
    print('STRADA-SMSE:')

    preds_strada_smse =\
        np.logical_or(preds_clustering,
                        preds_l2_dist_smse)

    get_scores_thresholded(preds_strada_smse,
                                        label)

    # get_scores_thresholded_by_category(preds_clustering,
    #                                     preds_l2_dist_smse_by_category,
    #                                     label)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Uneduced HLT Dataset Evaluation')

    parser.add_argument('--data-dir', type=str, default='../../datasets/hlt')
    parser.add_argument('--seed', type=int)
  
    args = parser.parse_args()

    labels_pd = pd.read_hdf(args.data_dir +\
                            '/unreduced_hlt_test_set_y.h5')

    labels_np = labels_pd.to_numpy()

    print_results(labels_np, args.seed)
