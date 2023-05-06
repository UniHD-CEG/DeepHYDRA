import os
import csv

import numpy as np
from scipy import stats
from scipy import special
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *

from src.spot import SPOT
from src.constants import *
from src.prediction_analysis import *


anomaly_categories = {'Point Global': 0b0000001,
                        'Point Contextual': 0b0000010,
                        'Persistent Global': 0b0000100,
                        'Persistent Contextual': 0b0001000,
                        'Collective Global': 0b0010000,
                        'Collective Trend': 0b0100000}


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    # try:
    #     roc_auc = roc_auc_score(actual, predict)
    # except:
    #     roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN #, roc_auc


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


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-3, level=0.02):
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
    lms = lm[0]

    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    label = label[:len(score)]

    pred = np.zeros_like(score, dtype=np.uint8)

    pred[ret['alarms']] = 1

    pred = adjust_predicts(pred, label, 0.1)

    mcc = matthews_corrcoef(label, pred)

#     mean_shift, var_shift = get_mean_and_var_shift(score, label)
# 
#     prob_pred = estimate_prediction_probability(score > pot_th,  label > 0.1)

    latencies, mean_latency, var_latency =\
                get_detection_latencies(score > pot_th,  label > 0.1)


    # pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)

    p_t = calc_point2point(pred, label)

    try:
        auroc = roc_auc_score(label, score)
    except:
         auroc = 0


    # print('POT result: ', p_t, pot_th, p_latency)

    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': auroc,
        'MCC': mcc,
        'threshold': pot_th,
        'mean latency': mean_latency,
        'latency var': var_latency,
    }, np.array(pred), latencies


def save_metrics_to_csv(filename,
                            row_name,
                            pred_train,
                            pred_test,
                            true,
                            q=1e-5,
                            level=0.02):

    true = true[:len(pred_test)]

    try:
        auroc = roc_auc_score(true, pred_test)
    except:
        auroc = 0

    lms = lm[0]
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
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    mean_shift, var_shift = get_mean_and_var_shift(pred_test, true)

    prob_pred = estimate_prediction_probability(pred_test > pot_th,  true > 0.1)

    latencies, mean_latency, var_latency =\
                get_detection_latencies(pred_test > pot_th,  true > 0.1)

    pred, p_latency = adjust_predicts(pred_test, true, pot_th, calc_latency=True)

    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))

    p_t = calc_point2point(pred, true)

    # print('POT result: ', p_t, pot_th, p_latency)

    mcc = matthews_corrcoef(true, pred)

    if not os.path.isfile(filename):
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

            header = ['Name',
                        'AUROC',
                        'F1',
                        'MCC',
                        'Precision',
                        'Recall'
                        'Threshold']

            csv_writer.writerow(header)

    with open(filename, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([row_name,
                                auroc,
                                p_t[0],
                                mcc,
                                p_t[1],
                                p_t[2],
                                pot_th])


def print_metric_comparison_by_categories(pred_train,
                                                pred_test,
                                                true,
                                                categories,
                                                q=1e-5,
                                                level=0.02):

    true = true[:len(pred_test)]
    categories = categories[:len(pred_test)]

    try:
        auroc = roc_auc_score(true, pred_test)
    except:
        auroc = 0

    lms = lm[0]
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
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])

    pred_adjusted_transformer =\
                adjust_predicts(pred_test,
                                    true,
                                    pot_th)

    print('\nBy Category:\n')

    for category, flag in anomaly_categories.items():
        print(category)

        mask = np.where(categories & flag, 1, 0)

        mask = np.logical_or(mask,
                    np.where(categories == 0, 1, 0))

        f1 = f1_score(true[mask],
                        pred_adjusted_transformer[mask])

        precision = precision_score(true[mask],
                        pred_adjusted_transformer[mask])

        recall = recall_score(true[mask],
                        pred_adjusted_transformer[mask])

        mcc = matthews_corrcoef(true[mask],
                    pred_adjusted_transformer[mask])

        p = estimate_prediction_probability(
                                pred_test[mask] > pot_th,
                                true[mask])

        print(f'\n\tPrecision: {precision:.3f}'
                f' Recall: {recall:.3f}'
                f' F1 score: {f1:.3f}'
                f' MCC: {mcc:.3f}'
                f' p: {p:.5f}')
                

    f1 = f1_score(true, pred_adjusted_transformer)

    precision = precision_score(true,
                    pred_adjusted_transformer)

    recall = recall_score(true,
                    pred_adjusted_transformer)

    mcc = matthews_corrcoef(true,
                pred_adjusted_transformer)

    p = estimate_prediction_probability(
                            pred_test > pot_th,
                            true)
        

    print('\nAll categories:\n')

    print(f'2nd Stage w/o Clustering'
            f'\n\tPrecision: {precision:.3f}'
            f' Recall: {recall:.3f}'
            f' F1 score: {f1:.3f}'
            f' MCC: {mcc:.3f}'
            f' for threshold {pot_th:.10f}'
            f'\n\nAUROC: {auroc:.3f}'
            f' p: {p:.5f}')
