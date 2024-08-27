#!/usr/bin/env python3

import argparse
import sys
import datetime as dt
import json
import logging

import numpy as np
import pandas as pd
import pylikwid
from tqdm import tqdm
from tqdm.contrib import tzip
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.append('../../')

from clustering.dbscananomalydetector import HLTDBSCANAnomalyDetector
from reduction.medianstdreducer import MedianStdReducer
from transformer_based_detection.informers.informerrunner import InformerRunner
from utils.offlinepbeastdataloader import OfflinePBeastDataLoader
from utils.anomalyregistry import JSONAnomalyRegistry
from utils.reduceddatabuffer import ReducedDataBuffer
from utils.exceptions import NonCriticalPredictionException
from utils.consolesingleton import ConsoleSingleton



# run_endpoints = [1404,
#                     8928,
#                     19296,
#                     28948]
# 
# channels_to_delete_last_run = [1357,
#                                 3685,
#                                 3184]


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='T-DBSCAN/Informer Offline HLT Anomaly Detection')

    parser.add_argument('--model', type=str, choices=['Informer-MSE', 'Informer-SMSE'])
    parser.add_argument('--checkpoint-dir', type=str, default='../../../transformer_based_detection')
    parser.add_argument('--data-dir', type=str, default='../../../datasets/hlt/')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--log-level', type=str, default='info')
    parser.add_argument('--log-dir', type=str, default='./log/')
    
    parser.add_argument('--dbscan-eps', type=float, default=3)
    parser.add_argument('--dbscan-min-samples', type=int, default=4)
    parser.add_argument('--dbscan-duration-threshold', type=int, default=4)

    parser.add_argument('--variant', type=str, choices=['2018', '2022', '2023'], default='2018')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_model_name = args.model.lower().replace('-', '_')

    log_filename = f'{args.log_dir}/strada_{log_model_name}'\
                            f'benchmark_log_{time_now_string}.log'

    logging_format = '[%(asctime)s] %(levelname)s: %(name)s: %(message)s'

    logger = logging.getLogger(__name__)

    logging.getLogger().addFilter(lambda record: 'running accelerated version on CPU' not in record.msg)

    logging.basicConfig(filename=log_filename,
                                    filemode='w',
                                    level=args.log_level.upper(),
                                    format=logging_format,
                                    datefmt='%Y-%m-%d %H:%M:%S')

    offline_pbeast_data_loader = OfflinePBeastDataLoader(
                                    '../../../../atlas-data-summary-runs-2018.html')

    run_numbers = offline_pbeast_data_loader.get_run_numbers()

    for run_number in run_numbers:

        logger.info(f'Starting data loading for run {run_number}')

        hlt_data_pd = offline_pbeast_data_loader[run_number]

        hlt_data_pd.index = _remove_timestamp_jumps(
                                pd.DatetimeIndex(hlt_data_pd.index))
        
        # hlt_data_pd = hlt_data_pd.iloc[10000:20000, :]

        rack_config = '2018' if args.variant in ['2018', '2022'] else '2023'

        median_std_reducer = MedianStdReducer(rack_config)
        
        informer_runner = InformerRunner(args.checkpoint_dir)

        tpu_labels = list(hlt_data_pd.columns.values)

        logger.info('Instantiating T-DBSCAN detector with parameters'
                                        f'DBSCAN ε: {args.dbscan_eps} '
                                        f'min_samples: {args.dbscan_min_samples} '
                                        f'duration threshold: {args.dbscan_duration_threshold}')

        dbscan_anomaly_detector =\
            HLTDBSCANAnomalyDetector(tpu_labels,
                                        args.dbscan_eps,
                                        args.dbscan_min_samples,
                                        args.dbscan_duration_threshold)

        logger.info('Successfully instantiated T-DBSCAN detector')
        logger.info(f'Instantiating model {args.model}')

        if args.model == 'Informer-SMSE':
            reduced_data_buffer = ReducedDataBuffer(size=65)

        else:
            reduced_data_buffer = ReducedDataBuffer(size=17)

        logger.info(f'Successfully instantiated model {args.model}')

        reduced_data_buffer.set_buffer_filled_callback(informer_runner.detect)
        
        json_anomaly_registry =\
                JSONAnomalyRegistry(args.output_dir)

        dbscan_anomaly_detector.register_detection_callback(
                        json_anomaly_registry.clustering_detection)
        informer_runner.register_detection_callback(
                        json_anomaly_registry.transformer_detection)

        timestamps = list(hlt_data_pd.index)
        hlt_data_np = hlt_data_pd.to_numpy()

        logger.info(f'Starting combined detection on data of run {run_number}')

        with logging_redirect_tqdm():
            for count, (timestamp, data) in enumerate(tzip(timestamps, hlt_data_np)):
                try:
                    dbscan_anomaly_detector.process(timestamp, data)

                    output_slice =\
                        median_std_reducer.reduce_numpy(tpu_labels,
                                                            timestamp,
                                                            data)
                    
                except NonCriticalPredictionException:
                    break

        logger.info(f'Processing of data of run {run_number} finished')

        log_file_name = f'run_{run_number}'

        json_anomaly_registry.write_log_file(log_file_name)

        logger.info(f'Exported results for run {run_number} '
                            f'to file {log_file_name}.json')
