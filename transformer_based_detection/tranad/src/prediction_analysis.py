import numpy as np
from scipy import stats
from scipy import special
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *


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
        return np.array([]), np.array([])

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


def get_mean_and_var_shift(predict, actual):

    predict = predict.astype(np.uint8)

    run_starts, run_ends = _get_anomalous_runs(actual)

    anomaly_period_mask = np.zeros(len(actual), dtype=np.uint8)

    for run_start, run_end in zip(run_starts, run_ends):
        anomaly_period_mask[run_start:run_end] = 1

    normal_period_mask = np.logical_not(anomaly_period_mask)

    predictions_during_normal_periods =\
        np.ma.masked_array(predict, mask=anomaly_period_mask)

    predictions_during_actual_anomaly_periods =\
        np.ma.masked_array(predict, mask=normal_period_mask)

    var_normal = predictions_during_normal_periods.var()
    mean_normal = predictions_during_normal_periods.mean()

    var_anomaly = predictions_during_actual_anomaly_periods.var()
    mean_anomaly = predictions_during_actual_anomaly_periods.mean()

    return var_anomaly - var_normal, mean_anomaly - mean_normal


def estimate_prediction_probability(predict, actual):
    predict = predict.astype(np.uint8)

    true_positive_window_starts,\
                true_positive_window_ends =\
                        _get_anomalous_runs(actual)

    p = np.count_nonzero(predict)/len(predict)

    probs_if_uniform_dist_true_positive =\
                [0]*len(true_positive_window_starts)

    prob_pred_anomalies_detected_if_uniform_dist = 1

    true_negative_window_starts = []
    true_negative_window_ends = []

    true_positive_window_end_last = 0

    # Compute probabilities for false positive and true positive windows

    for index, (window_start, window_end) in\
                    enumerate(zip(true_positive_window_starts,
                                            true_positive_window_ends)):

        true_negative_window_starts.append(
                        true_positive_window_end_last)

        true_negative_window_ends.append(window_start)

        true_positive_window_end_last = window_end

        run_length = window_end - window_start
        probs_if_uniform_dist_true_positive[index] =\
                                    1 - (1 - p)**run_length

        if np.count_nonzero(predict[window_start:window_end]):
            prob_pred_anomalies_detected_if_uniform_dist *=\
                            probs_if_uniform_dist_true_positive[index]
        else:

            prob_pot = 0

            for count in range(1, run_length + 1):

                # Check if probability computation without multiplication
                # with the the binomial coefficient is zero. This avoids
                # multiplying inf as a result of special.comb(run_length, count)
                # with a zero, resulting in a NaN

                prob_no_binomial_coefficient =\
                        p**count*(1 - p)**(run_length - count + 1)

                if prob_no_binomial_coefficient != 0:
                    prob = special.comb(run_length, count)*\
                                    prob_no_binomial_coefficient

                else:
                    prob = 0

                prob_pot += prob

            prob_no_pot = 1 - prob_pot

            prob_pred_anomalies_detected_if_uniform_dist*=\
                                                    prob_no_pot

    probs_if_uniform_dist_true_negative =\
                [0]*len(true_negative_window_starts)

    for index, (window_start, window_end) in\
                    enumerate(zip(true_negative_window_starts,
                                            true_negative_window_ends)):
        
        run_length = window_end - window_start
        
        false_positives =\
            np.count_nonzero(predict[window_start:window_end])

        # Check if probability computation without multiplication
        # with the the binomial coefficient is zero. This avoids
        # multiplying inf as a result of special.comb(run_length, false_positives)
        # with a zero, resulting in a NaN

        prob_no_binomial_coefficient =\
            p**false_positives*(1 - p)**(run_length - false_positives)

        if prob_no_binomial_coefficient != 0:
            prob = special.comb(run_length, false_positives)*\
                                    prob_no_binomial_coefficient

        else:
            prob = 0

        probs_if_uniform_dist_true_negative[index] = prob
        
        prob_pred_anomalies_detected_if_uniform_dist *=\
                    probs_if_uniform_dist_true_negative[index]

    return prob_pred_anomalies_detected_if_uniform_dist


def get_detection_latencies(predict, actual):

    if len(predict) != len(actual):
        raise ValueError("score and label must have the same length")

    predict = np.asarray(predict)
    actual = np.asarray(actual)

    # print(predict.shape)
    # print(actual.shape)

    true_positive_window_starts,\
                true_positive_window_ends =\
                        _get_anomalous_runs(actual)

    latencies = np.full(len(true_positive_window_starts), float('nan'))

    for index, (window_start, window_end) in\
                enumerate(zip(true_positive_window_starts,
                                        true_positive_window_ends)):

        pots = np.flatnonzero(predict[window_start:window_end])

        # print(f'[{window_start}:{window_end}]: {len(pots)}')

        if len(pots):
            latencies[index] = pots[0]

    mean_latency = np.nanmean(latencies)
    var_latency = np.nanvar(latencies)

    return latencies, mean_latency, var_latency