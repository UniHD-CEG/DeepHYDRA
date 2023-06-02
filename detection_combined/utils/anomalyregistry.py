#!/usr/bin/env python3

import json
from collections import defaultdict
from abc import ABC

import numpy as np
import pandas as pd

from .anomalyclassification import RunAnomaly, AnomalyType


base_data_anomaly_starts = [247,
                                465,
                                4272]

base_data_anomaly_ends = [264,
                            465,
                            4277]

output_dir = '../../../evaluation/combined_detection_2022/predictions/'

def _save_numpy_array(array: np.array,
                        filename: str):
    with open(filename, 'wb') as output_file:
        np.save(output_file, array)


def _remove_timestamp_jumps(index: pd.DatetimeIndex) -> pd.DatetimeIndex:

    # Replace large timestamp jumps resulting from
    # consecutive datapoints coming from different
    # runs with a delta of 5 s, which is the average
    # update frequency of L1 rate data

    delta = index[1:] - index[:-1]

    index = pd.Series(index)

    for i in range(1, len(index)):
        if delta[i - 1] >= pd.Timedelta(10, unit='s'):

            index[i:] = index[i:] - delta[i - 1] +\
                            pd.Timedelta(5, unit='s')

    index = pd.DatetimeIndex(index)

    assert index.is_monotonic_increasing

    return index


class AnomalyRegistry(ABC):
    pass


class BenchmarkAnomalyRegistry(AnomalyRegistry):

    def __init__(self,
                    log_dir: str,
                    label_dir: str) -> None:
 
        self.log_dir = log_dir
        self.label_dir = label_dir
        self.anomaly_registry_persistent =\
                defaultdict(lambda: defaultdict(RunAnomaly))


    def clustering_detection(self,
                                origin_code: int,
                                anomaly_type: AnomalyType,
                                anomaly_start,
                                anomaly_duration: int) -> None:
        
        # Artificially limit durations reported
        # by the first stage detector to reduce
        # influence of actual dropouts on the
        # benchmark anomaly labeling

        self.anomaly_registry_persistent[origin_code][anomaly_start].update(anomaly_duration,
                                                                                    anomaly_type)

    def transformer_detection(self,
                                origin_code: int,
                                anomaly_type: AnomalyType,
                                anomaly_start,
                                anomaly_duration: int) -> None:
        
        self.anomaly_registry_persistent[origin_code][anomaly_start].update(anomaly_duration,
                                                                                    anomaly_type)


    def evaluate(self,
                    pred_transformer_np: np.array,
                    model_name: str,
                    variant: str,
                    seed: int) -> None:

        true_pd = pd.read_hdf(self.label_dir +\
                                    f'/unreduced_hlt_test_set_{variant}_y.h5')

        true_pd.index = _remove_timestamp_jumps(
                            pd.DatetimeIndex(true_pd.index)).strftime('%Y-%m-%d %H:%M:%S')

        true_np = np.any(true_pd.to_numpy()>=1, axis=1).astype(np.uint8).flatten()

        base_data_anomalies_np = np.zeros_like(true_np)

        for start, end in zip(base_data_anomaly_starts,
                                    base_data_anomaly_ends):
            base_data_anomalies_np[start:end] = 1

        true_np = np.logical_or(true_np, base_data_anomalies_np)

        pred_pd = pd.DataFrame(np.zeros_like(true_np), true_pd.index)

        # Build predictions from first stage anomaly detection registry entries

        for anomaly_origin, anomaly_starts in self.anomaly_registry_persistent.items():
            for anomaly_start, anomaly in anomaly_starts.items():
                if anomaly_origin != 0:
                    start = pred_pd.index.get_loc(anomaly_start)

                    for row in range(start, start + anomaly.duration):
                        pred_pd.iloc[row] = 1
                
        pred_clustering_np = pred_pd.to_numpy().astype(np.uint8).flatten()
        _save_numpy_array(pred_clustering_np, f'{output_dir}/clustering.npy')

        if model_name == 'Informer-MSE':
            pred_transformer_np = np.pad(pred_transformer_np, (16, 1))
            _save_numpy_array(pred_transformer_np, f'{output_dir}/l2_dist_mse_seed_{seed}.npy')
        elif model_name == 'Informer-SMSE':
            pred_transformer_np = np.pad(pred_transformer_np, (64, 1))
            _save_numpy_array(pred_transformer_np, f'{output_dir}/l2_dist_smse_seed_{seed}.npy')
        elif model_name == 'TranAD':
            pred_transformer_np = np.pad(pred_transformer_np, (9, 0))
            _save_numpy_array(pred_transformer_np, f'{output_dir}/tranad_seed_{seed}.npy')