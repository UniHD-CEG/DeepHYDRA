import os
import csv

import numpy as np

# from sklearnex import patch_sklearn
# patch_sklearn(verbose=False)

from sklearn.metrics import auc,\
                                f1_score,\
                                precision_score,\
                                recall_score,\
                                precision_recall_curve,\
                                precision_recall_fscore_support,\
                                matthews_corrcoef,\
                                roc_auc_score

import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

from .prediction_analysis import *

image_width = 1920
image_height = 1080

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 1
line_type = 2

font_color = (0, 0, 0)

red = '#e41a1cf0'
green = '#4daf4af0'
white = '#fffffff0'

anomaly_categories = {'Point Global': 0b0000001,
                        'Point Contextual': 0b0000010,
                        'Persistent Global': 0b0000100,
                        'Persistent Contextual': 0b0001000,
                        'Collective Global': 0b0010000,
                        'Collective Trend': 0b0100000,
                        'Intra Rack': 0b1000000}

def _fig_to_numpy_array(fig):
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


def _get_anomalous_runs(x):
    """Find runs of consecutive items in an array.
        As published in https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065"""

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


def sequence_precision_delay(output, true, delta_max, steps=100):

    thresholds = np.linspace(0, 1, steps)

    nadd_over_thresholds = np.empty_like(thresholds)
    precision_over_thresholds = np.empty_like(thresholds)
    recall_over_thresholds = np.empty_like(thresholds)

    anomaly_starts_actual, anomaly_ends_actual = _get_anomalous_runs(true)

#     four_cc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
# 
#     writer = cv.VideoWriter('detections_over_thresholds.mp4',
#                                                 four_cc, 25,
#                                                 (image_width,
#                                                     image_height))

    # for threshold_count, threshold in enumerate(tqdm(thresholds)):
    for threshold_count, threshold in enumerate(thresholds):

        pred = np.where(output >= threshold, 1, 0)

        anomaly_starts_pred, ends_pred = _get_anomalous_runs(pred)

#         anomaly_coloring = []
# 
#         for anomaly_actual, anomaly_prediction in zip(true, pred):
#             if anomaly_actual == 1:
#                 if anomaly_prediction == 1:
#                     anomaly_coloring.append(green)
#                 else:
#                     anomaly_coloring.append(red)
#             else:
#                 if anomaly_prediction == 1:
#                     anomaly_coloring.append(red)
#                 else:
#                     anomaly_coloring.append(white)
# 
#         fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=240)
# 
#         ax.set_ylim(-0.1, 1)
# 
#         ax.axes.get_yaxis().set_visible(False)
# 
#         ax.set_xlabel("Timestep")
#         ax.set_ylabel("s_t")

#         ax.bar(np.arange(len(output)),
#                     np.full_like(output, 1.1),
#                     width=-1,
#                     bottom=-0.1,
#                     align='edge',
#                     color=anomaly_coloring,
#                     zorder=0)
# 
#         ax.axhline(y=threshold, color='k', linestyle='--', zorder=10)
# 
#         ax.plot(np.arange(len(output)), output, 'k', zorder=10)
# 
#         frame = _fig_to_numpy_array(fig)
# 
#         cv.putText(frame,
#                     f'Threshold: {threshold:.3f}',
#                     (20,  10),
#                     font,
#                     font_scale,
#                     font_color,
#                     thickness,
#                     line_type)
# 
#         writer.write(frame)
# 
#         plt.close()

        detection_delays = np.full_like(anomaly_starts_actual, delta_max)

        anomaly_window_buckets = [set()]*len(anomaly_starts_actual)

        true_positives = 0

        for index_anomalous_run,\
                (anomaly_start_actual,\
                anomaly_end_actual) in enumerate(zip(anomaly_starts_actual,
                                                        anomaly_ends_actual)):
            
            detection_time_latest = min(anomaly_end_actual,
                                            anomaly_start_actual + delta_max)

            anomalies_pred_in_range =\
                anomaly_starts_pred[anomaly_starts_pred >= anomaly_start_actual]
            anomalies_pred_in_range =\
                anomalies_pred_in_range[anomalies_pred_in_range < detection_time_latest]

            if len(anomalies_pred_in_range):
                detection_delays[index_anomalous_run] =\
                                anomalies_pred_in_range[0] - anomaly_start_actual

                anomaly_window_buckets[index_anomalous_run].update(anomalies_pred_in_range)

                true_positives += 1

        anomaly_starts_pred = anomaly_starts_pred.tolist()

        for anomaly_window_bucket in anomaly_window_buckets:
            for anomaly_start_pred in anomaly_window_bucket:
                if anomaly_start_pred in anomaly_starts_pred:
                    anomaly_starts_pred.remove(anomaly_start_pred)
                
        false_positives = len(anomaly_starts_pred)
        positives_actual = len(anomaly_starts_actual)

        predicted_positives = true_positives + false_positives

        precision = true_positives/predicted_positives \
                            if predicted_positives > 0 else 0

        recall = true_positives/positives_actual

        # print(f'PR: {precision:.3f}\tREC: {recall:.3f}')

        nadd = np.mean(detection_delays/delta_max)

        nadd_over_thresholds[threshold_count] = nadd

        precision_over_thresholds[threshold_count] = precision
        recall_over_thresholds[threshold_count] = recall

    # writer.release()

    desc_score_indices = np.argsort(nadd_over_thresholds,
                                                kind="mergesort")

    nadd_over_thresholds = nadd_over_thresholds[desc_score_indices]
    precision_over_thresholds = precision_over_thresholds[desc_score_indices]

    spd = auc(nadd_over_thresholds, precision_over_thresholds)

    return spd, precision_over_thresholds, nadd_over_thresholds


def get_max_f1_score_by_pr_curve(pred, true):

    precision, recall, thresholds =\
                precision_recall_curve(true, pred)

    f1_scores = 2*precision*recall/(precision + recall)

    index_max_f1_score = np.nanargmax(f1_scores)

    max_f1_score = f1_scores[index_max_f1_score]

    precision_max_f1_score = precision[index_max_f1_score]
    recall_max_f1_score = recall[index_max_f1_score]
    threshold_max_f1_score = thresholds[index_max_f1_score]

    pred_thresholded =\
        np.where(pred >= threshold_max_f1_score, 1, 0)

    mcc_max_f1_score =\
        matthews_corrcoef(true, pred_thresholded)

    return max_f1_score,\
            precision_max_f1_score,\
            recall_max_f1_score,\
            mcc_max_f1_score,\
            threshold_max_f1_score

def get_max_f1_score_by_spot(pred_train, pred_test, true, q=1e-3, level=0.8):
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
                    verbose=True)

    # run
    ret = spot.run()

    print(len(ret['alarms']))
    print(len(ret['thresholds']))

    pred = np.zeros_like(pred_test, dtype=np.uint8)

    pred[ret['alarms']] = 1

    pred = adjust_predicts(pred, true, 0.1)

    precision_max_f1_score,\
        recall_max_f1_score,\
        max_f1_score, _ = precision_recall_fscore_support(true,
                                                            pred,
                                                            average='binary')

    mcc_max_f1_score =\
        matthews_corrcoef(true, pred)

    auroc = roc_auc_score(true, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {max_f1_score:.3f}\t'
            f'MCC: {mcc_max_f1_score:.3f}\t'
            f'Precision: {precision_max_f1_score:.3f}\t'
            f'Recall: {recall_max_f1_score:.3f}')

    return max_f1_score,\
            precision_max_f1_score,\
            recall_max_f1_score,\
            mcc_max_f1_score,\
            ret['thresholds']


def rse(pred, true):
    return np.sqrt(np.sum((true - pred)**2))/np.sqrt(np.sum((true - true.mean())**2))


def corr(pred, true):
    u = ((true - true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true - true.mean(0))**2*(pred - pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def mse(pred, true):
    return np.mean((pred - true)**2)


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mape(pred, true):
    return np.mean(np.abs((pred - true)/true))


def mspe(pred, true):
    return np.mean(np.square((pred - true)/true))


def metric(pred, true):
    return mae(pred, true),\
            mse(pred, true),\
            rmse(pred, true),\
            mape(pred, true),\
            mspe(pred, true)


# def print_metric_comparison_by_categories_fixed_threshold(pred_clustering,
#                                                             pred_transformer,
#                                                             true,
#                                                             categories,
#                                                             threshold,
#                                                             title):
# 
#     pred_adjusted_clustering =\
#             adjust_predicts(pred_clustering, true, 0.1)
#     
#     auroc = roc_auc_score(true, pred_transformer)
# 
#     pred_adjusted_transformer =\
#         adjust_predicts(pred_transformer,
#                             true,
#                             threshold)
# 
#     overlap = np.count_nonzero(
#                     np.logical_and(pred_adjusted_clustering,
#                                     pred_adjusted_transformer))/\
#                     len(pred_clustering)
# 
#     print(f'\nPrediction Method: {title}/DBSCAN\n')
# 
#     print(f'Prediction overlap: {overlap/100:.3f} %')
# 
#     print('\nBy Category:\n')
# 
#     for category, flag in anomaly_categories.items():
#         print(category)
# 
#         mask = np.where(categories & flag, 1, 0)
# 
#         mask = np.logical_or(mask,
#                     np.where(categories == 0, 1, 0))
# 
#         f1_clustering = f1_score(true[mask],
#                             pred_adjusted_clustering[mask])
# 
#         precision_clustering =\
#                     precision_score(true[mask],
#                             pred_adjusted_clustering[mask])
# 
#         recall_clustering =\
#                     recall_score(true[mask],
#                             pred_adjusted_clustering[mask])
# 
#         mcc_clustering =\
#             matthews_corrcoef(true[mask],
#                             pred_adjusted_clustering[mask])
#         
#         p_clustering =\
#             estimate_prediction_probability(
#                                     pred_clustering[mask], 
#                                     true[mask])
# 
#         print('Clustering Only'
#                 f'\n\tPrecision: {precision_clustering:.3f}'
#                 f' Recall: {recall_clustering:.3f}'
#                 f' F1 score: {f1_clustering:.3f}'
#                 f' MCC: {mcc_clustering:.3f}'
#                 f' p: {p_clustering:.3f}')
# 
#         f1_transformer = f1_score(true[mask],
#                         pred_adjusted_transformer[mask])
# 
#         precision_transformer =\
#                     precision_score(true[mask],
#                         pred_adjusted_transformer[mask])
# 
#         recall_transformer =\
#                     recall_score(true[mask],
#                         pred_adjusted_transformer[mask])
# 
#         mcc_transformer =\
#             matthews_corrcoef(true[mask],
#                     pred_adjusted_transformer[mask])
# 
#         p_transformer =\
#             estimate_prediction_probability(
#                                 pred_transformer[mask] >\
#                                                 threshold,
#                                 true[mask])
# 
#         print('2nd Stage w/o Clustering'
#                 f'\n\tPrecision: {precision_transformer:.3f}'
#                 f' Recall: {recall_transformer:.3f}'
#                 f' F1 score: {f1_transformer:.3f}'
#                 f' MCC: {mcc_transformer:.3f}'
#                 f' p: {p_transformer:.5f}')
# 
#         pred_combined =\
#                 np.logical_or(pred_adjusted_clustering[mask],
#                                 pred_adjusted_transformer[mask])
# 
#         f1_combined = f1_score(true[mask],
#                                 pred_combined)
# 
#         precision_combined =\
#                     precision_score(true[mask],
#                                         pred_combined)
# 
#         recall_combined =\
#                     recall_score(true[mask],
#                                     pred_combined)
# 
#         mcc_combined =\
#             matthews_corrcoef(true[mask],
#                                 pred_combined)
# 
#         p_combined =\
#             estimate_prediction_probability(
#                     np.logical_or(pred_clustering[mask], 
#                                     pred_transformer[mask] >\
#                                                     threshold),
#                                     true[mask])
# 
#         print('Combined'
#                 f'\n\tPrecision: {precision_combined:.3f}'
#                 f' Recall: {recall_combined:.3f}'
#                 f' F1 score: {f1_combined:.3f}'
#                 f' MCC: {mcc_combined:.3f}'
#                 f' p: {p_combined:.5f}\n')
# 
#     f1_clustering = f1_score(true,
#                             pred_adjusted_clustering)
# 
#     precision_clustering =\
#                 precision_score(true,
#                         pred_adjusted_clustering)
# 
#     recall_clustering =\
#                 recall_score(true,
#                         pred_adjusted_clustering)
# 
#     mcc_clustering =\
#         matthews_corrcoef(true,
#                         pred_adjusted_clustering)
# 
#     p_clustering =\
#         estimate_prediction_probability(pred_clustering,  true)
# 
#     print('\nAll categories:\n')
# 
#     print('Clustering Only'
#             f'\n\tPrecision: {precision_clustering:.3f}'
#             f' Recall: {recall_clustering:.3f}'
#             f' F1 score: {f1_clustering:.3f}'
#             f' MCC: {mcc_clustering:.3f}'
#             f' p: {p_clustering:.5f}')
# 
# 
#     f1_clustering =\
#             matthews_corrcoef(true,
#                     pred_adjusted_transformer)
# 
# 
#     precision_transformer =\
#                     precision_score(true,
#                         pred_adjusted_transformer)
# 
#     recall_transformer =\
#                     recall_score(true,
#                         pred_adjusted_transformer)
# 
#     mcc_transformer =\
#             matthews_corrcoef(true,
#                     pred_adjusted_transformer)
# 
# 
#     p_transformer =\
#         estimate_prediction_probability(
#                             pred_transformer >\
#                                 threshold,
#                             true)
# 
#     print(f'2nd Stage w/o Clustering'
#             f'\n\tPrecision: {precision_transformer:.3f}'
#             f' Recall: {recall_transformer:.3f}'
#             f' F1 score: {f1_clustering:.3f}'
#             f' MCC: {precision_transformer:.3f}'
#             f' for threshold {threshold:.3f}'
#             f'\n\nAUROC: {auroc:.3f}'
#             f' p: {p_transformer:.5f}')
# 
#     pred_combined =\
#                 np.logical_or(pred_adjusted_clustering,
#                                 pred_adjusted_transformer)
# 
#     f1_combined = f1_score(true,
#                             pred_combined)
# 
#     precision_combined =\
#                 precision_score(true,
#                                     pred_combined)
# 
#     recall_combined =\
#                 recall_score(true,
#                                 pred_combined)
# 
#     mcc_combined =\
#         matthews_corrcoef(true,
#                             pred_combined)
# 
#     p_combined =\
#         estimate_prediction_probability(
#                 np.logical_or(pred_clustering, 
#                                 pred_transformer >\
#                                         threshold),
#                                 true)
# 
#     print('Combined'
#             f'\n\tPrecision: {precision_combined:.3f}'
#             f' Recall: {recall_combined:.3f}'
#             f' F1 score: {f1_combined:.3f}'
#             f' MCC: {mcc_combined:.3f}'
#             f' p: {p_combined:.5f}')


def print_metric_comparison_by_categories(pred_clustering,
                                            pred_transformer,
                                            true,
                                            categories,
                                            title):

    pred_adjusted_clustering =\
            adjust_predicts(pred_clustering, true, 0.1)
    
    auroc = roc_auc_score(true, pred_transformer)

    mask = np.logical_not(np.where((categories & 0b01000000), 1, 0))

    # mask = np.logical_or(mask,
    #                 np.where(categories == 0, 1, 0))

    _, _, _, _, threshold_max_f1_score_pr_auc =\
                    get_max_f1_score_by_pr_curve(pred_transformer[mask], true[mask])

    _, _, _, _, threshold_max_f1_score_pr_auc =\
                    get_max_f1_score_by_pr_curve(preds_l2_dist, label)

    spot_train_size = int(len(preds_l2_dist)*0.1)

    preds_l2_dist =\
        np.pad(preds_l2_dist[1:], (0, 1),
                                'constant',
                                constant_values=(0,))

    get_max_f1_score_by_spot(preds_l2_dist_train[:spot_train_size],
                                            preds_l2_dist, label, 0.01)

    preds_l2_dist = preds_l2_dist >= threshold_max_f1_score_pr_auc

    pred_adjusted_transformer =\
        adjust_predicts(pred_transformer,
                            true,
                            threshold_max_f1_score_pr_auc)

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc, _ =\
            get_max_f1_score_by_pr_curve(pred_adjusted_transformer, true)
    
    overlap = np.count_nonzero(
                    np.logical_and(pred_adjusted_clustering,
                                    pred_adjusted_transformer))/\
                    len(pred_clustering)

    print(f'\nPrediction Method: {title}/DBSCAN\n')

    print(f'Prediction overlap: {overlap/100:.3f} %')

    print('\nBy Category:\n')

    for category, flag in anomaly_categories.items():
        print(category)

        mask = np.where(categories & flag, 1, 0)

        mask = np.logical_or(mask,
                    np.where(categories == 0, 1, 0))

        f1_clustering = f1_score(true[mask],
                            pred_adjusted_clustering[mask])

        precision_clustering =\
                    precision_score(true[mask],
                            pred_adjusted_clustering[mask])

        recall_clustering =\
                    recall_score(true[mask],
                            pred_adjusted_clustering[mask])

        mcc_clustering =\
            matthews_corrcoef(true[mask],
                            pred_adjusted_clustering[mask])
        
        p_clustering =\
            estimate_prediction_probability(
                                    pred_clustering[mask], 
                                    true[mask])

        print('Clustering Only'
                f'\n\tPrecision: {precision_clustering:.3f}'
                f' Recall: {recall_clustering:.3f}'
                f' F1 score: {f1_clustering:.3f}'
                f' MCC: {mcc_clustering:.3f}'
                f' p: {p_clustering:.3f}')

        f1_transformer = f1_score(true[mask],
                        pred_adjusted_transformer[mask])

        precision_transformer =\
                    precision_score(true[mask],
                        pred_adjusted_transformer[mask])

        recall_transformer =\
                    recall_score(true[mask],
                        pred_adjusted_transformer[mask])

        mcc_transformer =\
            matthews_corrcoef(true[mask],
                    pred_adjusted_transformer[mask])

        p_transformer =\
            estimate_prediction_probability(
                                pred_transformer[mask] >\
                                    threshold_max_f1_score_pr_auc,
                                true[mask])

        print('2nd Stage w/o Clustering'
                f'\n\tPrecision: {precision_transformer:.3f}'
                f' Recall: {recall_transformer:.3f}'
                f' F1 score: {f1_transformer:.3f}'
                f' MCC: {mcc_transformer:.3f}'
                f' p: {p_transformer:.5f}')

        pred_combined =\
                np.logical_or(pred_adjusted_clustering[mask],
                                pred_adjusted_transformer[mask])

        f1_combined = f1_score(true[mask],
                                pred_combined)

        precision_combined =\
                    precision_score(true[mask],
                                        pred_combined)

        recall_combined =\
                    recall_score(true[mask],
                                    pred_combined)

        mcc_combined =\
            matthews_corrcoef(true[mask],
                                pred_combined)

        p_combined =\
            estimate_prediction_probability(
                    np.logical_or(pred_clustering[mask], 
                                    pred_transformer[mask] >\
                                        threshold_max_f1_score_pr_auc),
                                    true[mask])

        print('Combined'
                f'\n\tPrecision: {precision_combined:.3f}'
                f' Recall: {recall_combined:.3f}'
                f' F1 score: {f1_combined:.3f}'
                f' MCC: {mcc_combined:.3f}'
                f' p: {p_combined:.5f}\n')

    f1_clustering = f1_score(true,
                            pred_adjusted_clustering)

    precision_clustering =\
                precision_score(true,
                        pred_adjusted_clustering)

    recall_clustering =\
                recall_score(true,
                        pred_adjusted_clustering)

    mcc_clustering =\
        matthews_corrcoef(true,
                        pred_adjusted_clustering)

    p_clustering =\
        estimate_prediction_probability(pred_clustering,  true)

    print('\nAll categories:\n')

    print('Clustering Only'
            f'\n\tPrecision: {precision_clustering:.3f}'
            f' Recall: {recall_clustering:.3f}'
            f' F1 score: {f1_clustering:.3f}'
            f' MCC: {mcc_clustering:.3f}'
            f' p: {p_clustering:.5f}')

    p_transformer =\
        estimate_prediction_probability(
                            pred_transformer >\
                                threshold_max_f1_score_pr_auc,
                            true)

    print(f'2nd Stage w/o Clustering'
            f'\n\tPrecision: {precision_max_f1_score_pr_auc:.3f}'
            f' Recall: {recall_max_f1_score_pr_auc:.3f}'
            f' F1 score: {max_f1_score_pr_auc:.3f}'
            f' MCC: {mcc_max_f1_score_pr_auc:.3f}'
            f' for threshold {threshold_max_f1_score_pr_auc:.3f}'
            f'\n\nAUROC: {auroc:.3f}'
            f' p: {p_transformer:.5f}')

    pred_combined =\
                np.logical_or(pred_adjusted_clustering,
                                pred_adjusted_transformer)

    f1_combined = f1_score(true,
                            pred_combined)

    precision_combined =\
                precision_score(true,
                                    pred_combined)

    recall_combined =\
                recall_score(true,
                                pred_combined)

    mcc_combined =\
        matthews_corrcoef(true,
                            pred_combined)

    p_combined =\
        estimate_prediction_probability(
                np.logical_or(pred_clustering, 
                                pred_transformer >\
                                    threshold_max_f1_score_pr_auc),
                                true)

    print('Combined'
            f'\n\tPrecision: {precision_combined:.3f}'
            f' Recall: {recall_combined:.3f}'
            f' F1 score: {f1_combined:.3f}'
            f' MCC: {mcc_combined:.3f}'
            f' p: {p_combined:.5f}')


def print_metric_comparison_by_categories_fixed_threshold(pred_clustering,
                                                            pred_transformer,
                                                            true,
                                                            categories,
                                                            threshold,
                                                            title):

    pred_adjusted_clustering =\
            adjust_predicts(pred_clustering, true, 0.1)
    
    auroc = roc_auc_score(true, pred_transformer)

    pred_adjusted_transformer =\
        adjust_predicts(pred_transformer,
                            true,
                            threshold)

    overlap = np.count_nonzero(
                    np.logical_and(pred_adjusted_clustering,
                                    pred_adjusted_transformer))/\
                    len(pred_clustering)

    print(f'\nPrediction Method: {title}/DBSCAN\n')

    print(f'Prediction overlap: {overlap/100:.3f} %')

    print('\nBy Category:\n')

    for category, flag in anomaly_categories.items():
        print(category)

        mask = np.where(categories & flag, 1, 0)

        mask = np.logical_or(mask,
                    np.where(categories == 0, 1, 0))

        f1_clustering = f1_score(true[mask],
                            pred_adjusted_clustering[mask])

        precision_clustering =\
                    precision_score(true[mask],
                            pred_adjusted_clustering[mask])

        recall_clustering =\
                    recall_score(true[mask],
                            pred_adjusted_clustering[mask])

        mcc_clustering =\
            matthews_corrcoef(true[mask],
                            pred_adjusted_clustering[mask])
        
        p_clustering =\
            estimate_prediction_probability(
                                    pred_clustering[mask], 
                                    true[mask])

        print('Clustering Only'
                f'\n\tPrecision: {precision_clustering:.3f}'
                f' Recall: {recall_clustering:.3f}'
                f' F1 score: {f1_clustering:.3f}'
                f' MCC: {mcc_clustering:.3f}'
                f' p: {p_clustering:.3f}')

        f1_transformer = f1_score(true[mask],
                        pred_adjusted_transformer[mask])

        precision_transformer =\
                    precision_score(true[mask],
                        pred_adjusted_transformer[mask])

        recall_transformer =\
                    recall_score(true[mask],
                        pred_adjusted_transformer[mask])

        mcc_transformer =\
            matthews_corrcoef(true[mask],
                    pred_adjusted_transformer[mask])

        p_transformer =\
            estimate_prediction_probability(
                                pred_transformer[mask] >\
                                                threshold,
                                true[mask])

        print('2nd Stage w/o Clustering'
                f'\n\tPrecision: {precision_transformer:.3f}'
                f' Recall: {recall_transformer:.3f}'
                f' F1 score: {f1_transformer:.3f}'
                f' MCC: {mcc_transformer:.3f}'
                f' p: {p_transformer:.5f}')

        pred_combined =\
                np.logical_or(pred_adjusted_clustering[mask],
                                pred_adjusted_transformer[mask])

        f1_combined = f1_score(true[mask],
                                pred_combined)

        precision_combined =\
                    precision_score(true[mask],
                                        pred_combined)

        recall_combined =\
                    recall_score(true[mask],
                                    pred_combined)

        mcc_combined =\
            matthews_corrcoef(true[mask],
                                pred_combined)

        p_combined =\
            estimate_prediction_probability(
                    np.logical_or(pred_clustering[mask], 
                                    pred_transformer[mask] >\
                                                    threshold),
                                    true[mask])

        print('Combined'
                f'\n\tPrecision: {precision_combined:.3f}'
                f' Recall: {recall_combined:.3f}'
                f' F1 score: {f1_combined:.3f}'
                f' MCC: {mcc_combined:.3f}'
                f' p: {p_combined:.5f}\n')

    f1_clustering = f1_score(true,
                            pred_adjusted_clustering)

    precision_clustering =\
                precision_score(true,
                        pred_adjusted_clustering)

    recall_clustering =\
                recall_score(true,
                        pred_adjusted_clustering)

    mcc_clustering =\
        matthews_corrcoef(true,
                        pred_adjusted_clustering)

    p_clustering =\
        estimate_prediction_probability(pred_clustering,  true)

    print('\nAll categories:\n')

    print('Clustering Only'
            f'\n\tPrecision: {precision_clustering:.3f}'
            f' Recall: {recall_clustering:.3f}'
            f' F1 score: {f1_clustering:.3f}'
            f' MCC: {mcc_clustering:.3f}'
            f' p: {p_clustering:.5f}')


    f1_transformer =\
            matthews_corrcoef(true,
                    pred_adjusted_transformer)


    precision_transformer =\
                    precision_score(true,
                        pred_adjusted_transformer)

    recall_transformer =\
                    recall_score(true,
                        pred_adjusted_transformer)

    mcc_transformer =\
            matthews_corrcoef(true,
                    pred_adjusted_transformer)


    p_transformer =\
        estimate_prediction_probability(
                            pred_transformer >\
                                threshold,
                            true)

    print(f'2nd Stage w/o Clustering'
            f'\n\tPrecision: {precision_transformer:.3f}'
            f' Recall: {recall_transformer:.3f}'
            f' F1 score: {f1_transformer:.3f}'
            f' MCC: {mcc_transformer:.3f}'
            f' for threshold {threshold:.3f}'
            f'\n\nAUROC: {auroc:.3f}'
            f' p: {p_transformer:.5f}')

    pred_combined =\
                np.logical_or(pred_adjusted_clustering,
                                pred_adjusted_transformer)

    f1_combined = f1_score(true,
                            pred_combined)

    precision_combined =\
                precision_score(true,
                                    pred_combined)

    recall_combined =\
                recall_score(true,
                                pred_combined)

    mcc_combined =\
        matthews_corrcoef(true,
                            pred_combined)

    p_combined =\
        estimate_prediction_probability(
                np.logical_or(pred_clustering, 
                                pred_transformer >\
                                        threshold),
                                true)

    print('Combined'
            f'\n\tPrecision: {precision_combined:.3f}'
            f' Recall: {recall_combined:.3f}'
            f' F1 score: {f1_combined:.3f}'
            f' MCC: {mcc_combined:.3f}'
            f' p: {p_combined:.5f}')


def print_metric_comparison(pred_clustering,
                                pred_transformer,
                                true,
                                title):

    pred_adjusted_clustering =\
            adjust_predicts(pred_clustering, true, 0.1)

    f1_clustering = f1_score(true, pred_clustering)

    precision_clustering =\
                precision_score(true, pred_adjusted_clustering)

    recall_clustering =\
                recall_score(true, pred_adjusted_clustering)

    mcc_clustering =\
        matthews_corrcoef(true, pred_adjusted_clustering)

    print('Clustering Only'
        f'\n\tPrecision: {precision_clustering:.3f}'
        f' Recall: {recall_clustering:.3f}'
        f' F1 score: {f1_clustering:.3f}'
        f' MCC: {mcc_clustering:.3f}')

    print_metrics(pred_transformer,
                        true,
                        f'{title} w/o Clustering')

    print_metrics_combined(pred_clustering,
                            pred_transformer,
                                        true,
                                        f'{title} with Clustering')

        

def print_metrics(pred,
                    true,
                    title,
                    scaler=None):

    # Print results without clustering

    auroc = roc_auc_score(true, pred)

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc,\
    threshold_max_f1_score_pr_auc =\
            get_max_f1_score_by_pr_curve(pred, true)

    pred_adjusted = adjust_predicts(pred, true, threshold_max_f1_score_pr_auc)

    _, mean_latency, latency_var =\
        get_detection_latencies(pred > threshold_max_f1_score_pr_auc,  true)

    var_shift, mean_shift = get_mean_and_var_shift(pred, true)

    p = estimate_prediction_probability(pred > threshold_max_f1_score_pr_auc,  true)

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc, _ =\
            get_max_f1_score_by_pr_curve(pred_adjusted, true)

    if scaler:
        threshold_max_f1_score_pr_auc =\
            scaler.inverse_transform([[threshold_max_f1_score_pr_auc]])[0][0]

    print(f'{title}'
            f'\n\tPrecision: {precision_max_f1_score_pr_auc:.3f}'
            f' Recall: {recall_max_f1_score_pr_auc:.3f}'
            f' F1 score: {max_f1_score_pr_auc:.3f}'
            f' MCC: {mcc_max_f1_score_pr_auc:.3f}'
            f' Mean latency: {mean_latency:.3f}'
            f' Latency var: {latency_var:.3f}'
            f' for threshold {threshold_max_f1_score_pr_auc:.3f}'
            f'\n\nAUROC: {auroc:.3f}'
            f' p: {p:.5f}'
            f' mean shift: {mean_shift:.3f}'
            f' var shift: {var_shift:.3f}')


def print_metrics_preadjusted(pred,
                                true,
                                title,
                                scaler=None):

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc,\
    threshold_max_f1_score_pr_auc =\
            get_max_f1_score_by_pr_curve(pred, true)

    pred_adjusted = adjust_predicts(pred, true, threshold_max_f1_score_pr_auc)

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc, _ =\
            get_max_f1_score_by_pr_curve(pred_adjusted, true)

    if scaler:
        threshold_max_f1_score_pr_auc =\
            scaler.inverse_transform([[threshold_max_f1_score_pr_auc]])[0][0]

    print(f'{title}'
            f'\n\tPrecision: {precision_max_f1_score_pr_auc:.3f}'
            f' Recall: {recall_max_f1_score_pr_auc:.3f}'
            f' F1 score: {max_f1_score_pr_auc:.3f}'
            f' MCC: {mcc_max_f1_score_pr_auc:.3f}')


def print_metrics_combined(pred_clustering,
                                pred_transformer,
                                true,
                                title,
                                scaler=None):

    # Print results without clustering

    auroc = roc_auc_score(true, pred_transformer)

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc,\
    threshold_max_f1_score_pr_auc =\
            get_max_f1_score_by_pr_curve(pred_transformer, true)

    pred_adjusted = adjust_predicts(pred_transformer, true, threshold_max_f1_score_pr_auc)

    pred_combined = np.logical_or(pred_clustering, pred_adjusted)

    _, mean_latency, latency_var =\
        get_detection_latencies(pred_transformer > threshold_max_f1_score_pr_auc,  true)

    var_shift, mean_shift = get_mean_and_var_shift(pred_transformer, true)

    p = estimate_prediction_probability(pred_transformer > threshold_max_f1_score_pr_auc,  true)

    max_f1_score_pr_auc,\
    precision_max_f1_score_pr_auc,\
    recall_max_f1_score_pr_auc,\
    mcc_max_f1_score_pr_auc, _ =\
        get_max_f1_score_by_pr_curve(pred_combined, true)

    if scaler:
        threshold_max_f1_score_pr_auc =\
            scaler.inverse_transform([[threshold_max_f1_score_pr_auc]])[0][0]

    print(f'{title}'
            f'\n\tPrecision: {precision_max_f1_score_pr_auc:.3f}'
            f' Recall: {recall_max_f1_score_pr_auc:.3f}'
            f' F1 score: {max_f1_score_pr_auc:.3f}'
            f' MCC: {mcc_max_f1_score_pr_auc:.3f}'
            f' Mean latency: {mean_latency:.3f}'
            f' Latency var: {latency_var:.3f}'
            f' for threshold {threshold_max_f1_score_pr_auc:.3f}'
            f'\n\nAUROC: {auroc:.3f}'
            f' p: {p:.5f}'
            f' mean shift: {mean_shift:.3f}'
            f' var shift: {var_shift:.3f}')


def save_metrics_to_csv(filename,
                            row_name,
                            pred,
                            true):

    auroc = roc_auc_score(true, pred)

    max_f1_score,\
    precision_max_f1_score,\
    recall_max_f1_score,\
    mcc_max_f1_score,\
    threshold_max_f1_score_pr_auc =\
            get_max_f1_score_by_pr_curve(pred, true)

    pred_adjusted = adjust_predicts(pred, true, threshold_max_f1_score_pr_auc)

    max_f1_score,\
    precision_max_f1_score,\
    recall_max_f1_score,\
    mcc_max_f1_score,\
    threshold_max_f1_score_pr_auc =\
            get_max_f1_score_by_pr_curve(pred_adjusted, true)


    if not os.path.isfile(filename):
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

            header = ['Name',
                        'AUROC',
                        'F1',
                        'MCC',
                        'Precision',
                        'Recall']

            csv_writer.writerow(header)

    with open(filename, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([row_name,
                                auroc,
                                max_f1_score,
                                mcc_max_f1_score,
                                precision_max_f1_score,
                                recall_max_f1_score])
