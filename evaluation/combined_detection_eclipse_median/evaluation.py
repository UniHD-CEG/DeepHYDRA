

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,\
                                precision_recall_fscore_support,\
                                matthews_corrcoef

sys.path.append('.')

from spot import SPOT


def load_numpy_array(filename: str):
    with open(filename, 'rb') as output_file:
        return np.load(output_file)


def save_to_csv(model_name: str,
                        seed: int,
                        auroc: np.float64,
                        f1: np.float64,
                        mcc: np.float64,
                        precision: np.float64,
                        recall: np.float64):
    
    metrics_to_save = [seed,
                        auroc,
                        f1, mcc,
                        precision,
                        recall]

    metrics_to_save = np.atleast_2d(metrics_to_save)

    metrics_to_save_pd = pd.DataFrame(data=metrics_to_save)
    metrics_to_save_pd.to_csv(f'results_combined_detection_{model_name}.csv',
                                                                    mode='a+',
                                                                    header=False,
                                                                    index=False)


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


def get_scores(model_name,
                seed,
                pred_train,
                pred_test,
                true,
                q=1e-3,
                level=0.8,
                to_csv=False):
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

    # pred = adjust_predicts(pred, true_reduced, 0.1)

    precision,\
        recall,\
        f1, _ = precision_recall_fscore_support(true_reduced,
                                                        pred,
                                                        average='binary')

    mcc =\
        matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {f1:.3f}\t'
            f'MCC: {mcc:.3f}\t'
            f'Precision: {precision:.3f}\t'
            f'Recall: {recall:.3f}')
    
    if to_csv:
        save_to_csv(model_name,
                        seed,
                        auroc,
                        f1,
                        mcc,
                        precision,
                        recall)

    return pred


def get_scores_tranad(model_name,
                            seed,
                            pred_train,
                            pred_test,
                            true,
                            q=1e-3,
                            level=0.02,
                            to_csv=False):
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
    pot_th = np.mean(ret['thresholds'])*3.5
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    pred = pred_test > pot_th

    # pred = adjust_predicts(pred, true_reduced, 0.1)

    precision,\
        recall,\
        f1, _ = precision_recall_fscore_support(true_reduced,
                                                        pred,
                                                        average='binary')

    mcc =\
        matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {f1:.3f}\t'
            f'MCC: {mcc:.3f}\t'
            f'Precision: {precision:.3f}\t'
            f'Recall: {recall:.3f}')
    
    if to_csv:
        save_to_csv(model_name,
                        seed,
                        auroc,
                        f1, mcc,
                        precision,
                        recall)

    return pred


def get_scores_dagmm(model_name,
                            seed,
                            pred_train,
                            pred_test,
                            true,
                            q=1e-3,
                            level=0.02,
                            thresh_tweak_factor=20,
                            to_csv=False):
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
    pot_th = np.mean(ret['thresholds'])*thresh_tweak_factor
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    pred = pred_test > pot_th

    # pred = adjust_predicts(pred, true_reduced, 0.1)

    precision,\
        recall,\
        f1, _ = precision_recall_fscore_support(true_reduced, pred,
                                                    average='binary')

    mcc = matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {f1:.3f}\t'
            f'MCC: {mcc:.3f}\t'
            f'Precision: {precision:.3f}\t'
            f'Recall: {recall:.3f}')
    
    if to_csv:
        save_to_csv(model_name,
                            seed,
                            auroc,
                            f1, mcc,
                            precision,
                            recall)

    return pred


def get_scores_thresholded(model_name,
                                    seed,
                                    pred,
                                    true,
                                    to_csv):

    true_reduced =\
        np.any(np.greater_equal(true, 1), axis=1).astype(np.uint8)

    precision,\
        recall,\
        f1, _ = precision_recall_fscore_support(true_reduced,
                                                        pred,
                                                        average='binary')

    mcc =\
        matthews_corrcoef(true_reduced, pred)

    auroc = roc_auc_score(true_reduced, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {f1:.3f}\t'
            f'MCC: {mcc:.3f}\t'
            f'Precision: {precision:.3f}\t'
            f'Recall: {recall:.3f}')
    
    if to_csv:
        save_to_csv(model_name,
                        seed,
                        auroc,
                        f1, mcc,
                        precision,
                        recall)


def print_results(label: np.array,
                        seed: int,
                        to_csv: bool):

    preds_method_3 =\
        load_numpy_array('predictions/method_3.npy')
    preds_method_4 =\
        load_numpy_array('predictions/method_4.npy')
    preds_clustering =\
        load_numpy_array('predictions/clustering.npy')
    preds_tranad_train =\
        load_numpy_array(f'predictions/tranad_train_no_augment_seed_{seed}.npy')
    preds_tranad =\
        load_numpy_array(f'predictions/tranad_no_augment_seed_{seed}.npy')
    preds_l2_dist_train_mse =\
        load_numpy_array(f'predictions/l2_dist_train_mse_no_augment_seed_{seed}.npy')
    preds_l2_dist_mse =\
        load_numpy_array(f'predictions/l2_dist_mse_no_augment_seed_{seed}.npy')
    preds_l2_dist_train_smse =\
        load_numpy_array(f'predictions/l2_dist_train_smse_no_augment_seed_{seed}.npy')
    preds_l2_dist_smse =\
        load_numpy_array(f'predictions/l2_dist_smse_no_augment_seed_{seed}.npy')

    preds_usad_train =\
        load_numpy_array(f'predictions/usad_train_no_augment_seed_{seed}.npy')    
    preds_usad =\
        load_numpy_array(f'predictions/usad_no_augment_seed_{seed}.npy')
    preds_omni_anomaly_train =\
        load_numpy_array(f'predictions/omnianomaly_train_no_augment_seed_{seed}.npy')
    preds_omni_anomaly =\
        load_numpy_array(f'predictions/omnianomaly_no_augment_seed_{seed}.npy')
    preds_dagmm_train =\
        load_numpy_array(f'predictions/dagmm_train_no_augment_seed_{seed}.npy')
    preds_dagmm =\
        load_numpy_array(f'predictions/dagmm_no_augment_seed_{seed}.npy')

    #spot_train_size = int(len(preds_l2_dist_mse)*0.2)
    spot_train_size = int(len(preds_l2_dist_train_mse)*0.25)

    preds_method_3 = np.sum(preds_method_3, axis=1)
    preds_method_4 = np.sum(preds_method_4, axis=1)

    preds_method_3 = np.squeeze(MinMaxScaler().fit_transform(np.atleast_2d(preds_method_3).T))
    preds_method_4 = np.squeeze(MinMaxScaler().fit_transform(np.atleast_2d(preds_method_4).T))

    # Fix alignment
    
    preds_l2_dist_mse =\
        np.pad(preds_l2_dist_mse,
                    (16, 77), 'constant',
                    constant_values=(1,))
    
    preds_l2_dist_smse =\
        np.pad(preds_l2_dist_smse,
                        (16, 77), 'constant',
                        constant_values=(1,))
    
    preds_tranad =\
        np.pad(preds_tranad,
                        (0, 93), 'constant',
                        constant_values=(1,))
    
    label_reduced =\
        np.any(np.greater_equal(label, 1), axis=1).astype(np.uint8)

    # preds_clustering =\
    #     adjust_predicts(preds_clustering,
    #                         label_reduced, 0.1)

    print('T-DBSCAN:')

    get_scores_thresholded('t_dbscan',
                                seed,
                                preds_clustering,
                                label,
                                to_csv)
    
    # print('TranAD:')

    preds_tranad = get_scores_tranad('tranad',
                                            seed,
                                            preds_tranad_train[spot_train_size:2*spot_train_size],
                                            preds_tranad,
                                            label, 0.001, 0.8,
                                            to_csv)
    
    print('DeepHYDRA-TranAD:')

    preds_strada_tranad =\
        np.logical_or(preds_clustering,
                            preds_tranad)
    
    get_scores_thresholded('deephydra_tranad',
                                    seed,
                                    preds_strada_tranad,
                                    label,
                                    to_csv)
    
    # print('Informer-MSE:')

    # preds_l2_dist_mse = get_scores('informer_mse', seed,
    #                                 preds_l2_dist_train_mse[spot_train_size:2*spot_train_size],
    #                                 preds_l2_dist_mse, label, 0.00001, 0.8, to_csv)

    # print('DeepHYDRA-MSE:')

    # preds_strada_mse =\
    #     np.logical_or(preds_clustering,
    #                     preds_l2_dist_mse)
    
    # get_scores_thresholded('deephydra_mse',
    #                                 seed,
    #                                 preds_strada_mse,
    #                                 label,
    #                                 to_csv)

    # print('Informer-SMSE:')

    # preds_l2_dist_smse = get_scores('informer_smse', seed,
    #                                     preds_l2_dist_train_smse[spot_train_size:2*spot_train_size],
    #                                     preds_l2_dist_smse, label, 0.0001, 0.8, to_csv)
    
    # print('DeepHYDRA-SMSE:')

    # preds_strada_smse =\
    #     np.logical_or(preds_clustering,
    #                     preds_l2_dist_smse)

    # get_scores_thresholded('deephydra_smse',
    #                                 seed,
    #                                 preds_strada_smse,
    #                                 label,
    #                                 to_csv)
    
    # print('DAGMM:')

    # preds_dagmm = get_scores_dagmm('dagmm', seed,
    #                                 preds_dagmm_train[spot_train_size:2*spot_train_size],
    #                                 preds_dagmm,
    #                                 label, 0.008,
    #                                 0.02, 10, to_csv)
    
    
    # print('DeepHYDRA-DAGMM:')

    # preds_strada_dagmm =\
    #     np.logical_or(preds_clustering,
    #                         preds_dagmm)

    # get_scores_thresholded('deephydra_dagmm',
    #                                 seed,
    #                                 preds_strada_dagmm,
    #                                 label,
    #                                 to_csv)
    
    # print('OmniAnomaly:')
    
    # preds_omni_anomaly = get_scores_dagmm('omnianomaly',
    #                                     seed,
    #                                     preds_omni_anomaly_train[spot_train_size:3*spot_train_size],
    #                                     preds_omni_anomaly,
    #                                     label, 0.001, 0.8, 1.6,
    #                                     to_csv)
    
    # print('DeepHYDRA-OmniAnomaly:')

    # preds_strada_omni_anomaly =\
    #     np.logical_or(preds_clustering,
    #                         preds_omni_anomaly)
    
    # get_scores_thresholded('deephydra_omnianomaly',
    #                                         seed,
    #                                         preds_strada_omni_anomaly,
    #                                         label,
    #                                         to_csv)
    
    print('USAD:')

    preds_usad = get_scores_dagmm('usad',
                                    seed,
                                    preds_usad_train[spot_train_size:2*spot_train_size],
                                    preds_usad,
                                    label, 0.1, 0.02, 18,
                                    to_csv)

    print('DeepHYDRA-USAD:')

    preds_strada_usad =\
        np.logical_or(preds_clustering,
                            preds_usad)
    
    get_scores_thresholded('deephydra_usad',
                                    seed,
                                    preds_strada_usad,
                                    label,
                                    to_csv)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unreduced Eclipse Dataset Evaluation')

    parser.add_argument('--data-dir', type=str, default='../../datasets/eclipse')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--to-csv', action='store_true', default=False)

    args = parser.parse_args()

    labels_pd = pd.read_hdf(args.data_dir +\
                            '/unreduced_eclipse_test_set.h5',
                            key='labels')

    labels_np = labels_pd.to_numpy()

    print_results(labels_np,
                    args.seed,
                    args.to_csv)
