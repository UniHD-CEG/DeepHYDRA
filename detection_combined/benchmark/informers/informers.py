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

from clustering.dbscananomalydetector import DBScanAnomalyDetector
from reduction.medianstdreducer import MedianStdReducer
from transformer_based_detection.informers.informerrunner import InformerRunner
from utils.anomalyregistry import BenchmarkAnomalyRegistry
from utils.reduceddatabuffer import ReducedDataBuffer
from utils.exceptions import NonCriticalPredictionException


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

    log_filename = f'{args.log_dir}/deep_hydra_{log_model_name}'\
                            f'benchmark_log_{time_now_string}.log'

    logging_format = '[%(asctime)s] %(levelname)s: %(name)s: %(message)s'

    logger = logging.getLogger(__name__)

    logging.getLogger().addFilter(lambda record: 'running accelerated version on CPU' not in record.msg)

    logging.basicConfig(filename=log_filename,
                                    filemode='w',
                                    level=args.log_level.upper(),
                                    format=logging_format,
                                    datefmt='%Y-%m-%d %H:%M:%S')

    hlt_data_pd = pd.read_hdf(args.data_dir +\
                                    f'/unreduced_hlt_dcm_test_set_{args.variant}_x.h5')

    # This removes a few actual anomalous dropouts in the last run.
    # These are very easy to detect, so we remove them to not
    # overshadow the the more subtle injected anomalies

    # hlt_data_pd.iloc[run_endpoints[-2]:-1,
    #                         channels_to_delete_last_run] = np.nan

    hlt_data_pd.index = _remove_timestamp_jumps(
                            pd.DatetimeIndex(hlt_data_pd.index))
    
    # hlt_data_pd = hlt_data_pd.iloc[10000:20000, :]

    rack_config = '2018' if args.variant in ['2018', '2022'] else '2023'

    median_std_reducer = MedianStdReducer(rack_config)
    
    informer_runner = InformerRunner(args.checkpoint_dir)

    tpu_labels = list(hlt_data_pd.columns.values)

    dbscan_anomaly_detector =\
        DBScanAnomalyDetector(tpu_labels,
                                args.dbscan_eps,
                                args.dbscan_min_samples,
                                args.dbscan_duration_threshold)

    
    if args.model == 'Informer-SMSE':
        reduced_data_buffer = ReducedDataBuffer(size=65)

    else:
        reduced_data_buffer = ReducedDataBuffer(size=17)

    reduced_data_buffer.set_buffer_filled_callback(informer_runner.detect)

    timestamps = list(hlt_data_pd.index)

    benchmark_anomaly_registry =\
            BenchmarkAnomalyRegistry(args.output_dir,
                                        args.data_dir)

    dbscan_anomaly_detector.register_detection_callback(
                    benchmark_anomaly_registry.clustering_detection)
    informer_runner.register_detection_callback(
                    benchmark_anomaly_registry.transformer_detection)

    hlt_data_np = hlt_data_pd.to_numpy()

    flops_dbscan = []
    flops_reduction = []

    with logging_redirect_tqdm():
        for count, (timestamp, data) in enumerate(tzip(timestamps, hlt_data_np)):

            # if count == 4096:
            #     dbscan_anomaly_detector.write_memory_size()
            #     break

            try:
                # pylikwid.markerinit()
                # pylikwid.markerthreadinit()

                # pylikwid.markerstartregion("DBSCAN")

                dbscan_anomaly_detector.process(timestamp, data)

                # pylikwid.markerstopregion("DBSCAN")

                # nr_events, eventlist, time, count = pylikwid.markergetregion("DBSCAN")

                # for i, e in enumerate(eventlist):
                #     print(i, e)

                # flops_dbscan.append(eventlist[3])

                # pylikwid.markerclose()

                # pylikwid.markerinit()
                # pylikwid.markerthreadinit()

                # pylikwid.markerstartregion("reduction")

                output_slice =\
                    median_std_reducer.reduce_numpy(tpu_labels,
                                                        timestamp,
                                                        data)
                                    
                # pylikwid.markerstopregion("reduction")

                # nr_events, eventlist, time, count = pylikwid.markergetregion("reduction")

                # flops_reduction.append(eventlist[3])

                # for i, e in enumerate(eventlist):
                #     print(i, e)
                # pylikwid.markerclose()
                
            except NonCriticalPredictionException:
                break

#     informer_runner.model.attention_visualizer.render_projection('smse_dcm_rate_data_2022_'\
#                                                                         'attention_viz_projection.mp4',
#                                                                     'Kevin Franz Stehle',
#                                                                     channels_upper=52,
#                                                                     fps=24,
#                                                                     label_size=20,
#                                                                     title_size=20,
#                                                                     cmap='plasma')
# 
#     informer_runner.model.attention_visualizer.render_combined('smse_dcm_rate_data_2022_'\
#                                                                     'attention_viz_combined.mp4',
#                                                                 'Kevin Franz Stehle',
#                                                                 fps=24,
#                                                                 label_size=20,
#                                                                 title_size=20,
#                                                                 cmap='plasma')
# 
#     informer_runner.model.attention_visualizer.render_individual_heads('smse_dcm_rate_data_2022_'\
#                                                                             'attention_viz_individual.mp4',
#                                                                         'Kevin Franz Stehle',
#                                                                         fps=24,
#                                                                         label_size=20,
#                                                                         title_size=20,
#                                                                         cmap='plasma')


    # flops = pd.DataFrame(np.column_stack((flops_dbscan, flops_reduction)).astype(np.uint64),
    #                                                         columns=['DBSCAN', 'Reduction'])

    # flops.to_csv('flops.csv')

    # print(flops_dbscan)
    # print(flops_reduction)

    # preds = informer_runner.get_predictions()

    # with open(args.checkpoint_dir +\
    #             '/model_parameters.json', 'r') as parameter_dict_file:
    #     parameter_dict = json.load(parameter_dict_file)

    #     benchmark_anomaly_registry.evaluate(preds,
    #                                             args.model,
    #                                             args.variant,
    #                                             args.seed)