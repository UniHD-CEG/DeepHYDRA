#!/usr/bin/env python3

from typing import List
import argparse
import sys
import time as t
import datetime as dt
import multiprocessing as mp
import asyncio as aio
import logging

import numpy as np
import pandas as pd

from rich.logging import RichHandler

sys.path.append('../../')

from clustering.dbscananomalydetector import DBScanAnomalyDetector
from reduction.medianstdreducer import MedianStdReducer
from transformer_based_detection.informers.informerrunner import InformerRunner
from utils.runcontrolstateprovider import RunControlStateProvider
from utils.onlinepbeastdataloader import OnlinePBeastDataLoader
from utils.anomalyregistry import JSONAnomalyRegistry
from utils.reduceddatabuffer import ReducedDataBuffer
from utils.exceptions import NonCriticalPredictionException
from utils.consolesingleton import ConsoleSingleton
from utils.gradioserver import GradioServer


# known_channels_2022 = ['m_1', 'm_2', 'm_3', 'm_4', 'm_5', 'm_6', 'm_7', 'm_8',
#                         'm_9', 'm_10', 'm_11', 'm_12', 'm_13', 'm_44', 'm_45', 'm_46',
#                         'm_47', 'm_48', 'm_49', 'm_50', 'm_51', 'm_52', 'm_53', 'm_54',
#                         'm_55', 'm_56', 'm_57', 'm_58', 'm_59', 'm_60', 'm_61', 'm_62',
#                         'm_63', 'm_64', 'm_65', 'm_66', 'm_67', 'm_68', 'm_69', 'm_70',
#                         'm_71', 'm_72', 'm_73', 'm_74', 'm_75', 'm_76', 'm_77', 'm_79',
#                         'm_80', 'm_81', 'm_82', 'm_83',
#                         'std_1', 'std_2', 'std_3', 'std_4', 'std_5', 'std_6', 'std_7', 'std_8',
#                         'std_9', 'std_10', 'std_11', 'std_12', 'std_13', 'std_44', 'std_45', 'std_46',
#                         'std_47', 'std_48', 'std_49', 'std_50', 'std_51', 'std_52', 'std_53', 'std_54',
#                         'std_55', 'std_56', 'std_57', 'std_58', 'std_59', 'std_60', 'std_61', 'std_62',
#                         'std_63', 'std_64', 'std_65', 'std_66', 'std_67', 'std_68', 'std_69', 'std_70',
#                         'std_71', 'std_72', 'std_73', 'std_74', 'std_75', 'std_76', 'std_77', 'std_79',
#                         'std_80', 'std_81', 'std_82', 'std_83']


# def fix_for_2023_deployment(data: pd.DataFrame) -> pd.DataFrame:
#     median_loc = int(np.flatnonzero(np.array(known_channels_2022) == 'm_63')[0])
#     std_loc = int(np.flatnonzero(np.array(known_channels_2022) == 'std_63')[0])

#     data = data.loc[:, data.columns.isin(known_channels_2022)]

#     data.insert(median_loc, 'm_63', data['m_62'].to_numpy())
#     data.insert(std_loc, 'std_63', data['std_62'].to_numpy())

#     return data


def polling_rate_parser(polling_rate_string: str):
    hours, minutes, seconds = map(int, polling_rate_string.split(':'))
    return dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)


async def wait_for_states(states: List[str],
                            return_delay: dt.timedelta):
    with console.status(f'Waiting for {states} transition...', spinner='simpleDots'):
        await run_control_state_provider.wait_for_states(states, return_delay)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='T-DBSCAN/Informer Offline HLT Anomaly Detection')

    parser.add_argument('--model', type=str, choices=['Informer-MSE', 'Informer-SMSE'])
    parser.add_argument('--cluster-configuration-version', type=str, choices=['2018', '2023'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint-dir', type=str, default='../../../transformer_based_detection')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--spot-based-detection', action='store_true', default=False)

    parser.add_argument('--cluster-state-polling-interval', type=str, default='00:00:10')

    parser.add_argument('--dbscan-eps', type=float, default=3)
    parser.add_argument('--dbscan-min-samples', type=int, default=4)
    parser.add_argument('--dbscan-duration-threshold', type=int, default=4)

    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--log-level', type=str, default='info')
    parser.add_argument('--log-dir', type=str, default='./log/')
    parser.add_argument('--anomaly-log-dump-interval', type=int, default=720)

    args = parser.parse_args()

    console = ConsoleSingleton().get_console()

    time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Configure logger

    log_model_name = args.model.lower().replace('-', '_')

    log_filename = f'{args.log_dir}/strada_{log_model_name}'\
                                    f'_log_{time_now_string}.log'

    logger = logging.getLogger()

    log_level = args.log_level.upper()

    logger.setLevel(log_level)
    logger.addFilter(lambda record: 'running accelerated version on CPU' not in record.msg)

    file_logging_format = '[%(asctime)s] %(levelname)s: %(name)s: %(message)s'

    file_logging_formatter = logging.Formatter(fmt=file_logging_format,
                                                datefmt='%Y-%m-%d %H:%M:%S')

    file_logging_handler = logging.FileHandler(log_filename, mode='w')
    file_logging_handler.setLevel(log_level)
    file_logging_handler.setFormatter(file_logging_formatter)

    console_logging_format = '%(name)s: %(message)s'

    console_logging_formatter = logging.Formatter(fmt=console_logging_format,
                                                    datefmt='%Y-%m-%d %H:%M:%S')

    console_logging_handler = RichHandler(console=console)
    console_logging_handler.setLevel(log_level)
    console_logging_handler.setFormatter(console_logging_formatter)

    logger.addHandler(file_logging_handler)
    logger.addHandler(console_logging_handler)

#     gradio_server = GradioServer()
# 
#     gradio_server_proc = mp.Process(target=gradio_server.launch)
#     gradio_server_proc.start()

    run_control_state_provider = RunControlStateProvider()

    aio.run(wait_for_states(['CONNECTED', 'RUNNING'],
                            dt.timedelta(seconds=5)))
    
    data_loader = OnlinePBeastDataLoader('DCMRate',
                                            polling_interval=dt.timedelta(seconds=5),
                                            delay=dt.timedelta(seconds=30),
                                            window_length=dt.timedelta(seconds=5),
                                            timing_violation_handling='skip')

    with console.status('Initializing dataloader', spinner='flip'):
        data_loader.init()

    tpu_labels = data_loader.get_column_names()

    median_std_reducer =\
            MedianStdReducer(args.cluster_configuration_version)

    loss_type = args.model.split('-')[-1].lower()
    
    with console.status('Loading model', spinner='flip'):
        informer_runner = InformerRunner(args.checkpoint_dir,
                                            loss_type=loss_type,
                                            use_spot_detection=args.spot_based_detection,
                                            device=args.device)

    dbscan_anomaly_detector =\
        DBScanAnomalyDetector(tpu_labels,
                                args.dbscan_eps,
                                args.dbscan_min_samples,
                                args.dbscan_duration_threshold)

    reduced_data_buffer_size = 65 if args.model == 'Informer-SMSE' else 17

    reduced_data_buffer = ReducedDataBuffer(size=reduced_data_buffer_size)

    reduced_data_buffer.set_buffer_filled_callback(informer_runner.detect)

    json_anomaly_registry =\
            JSONAnomalyRegistry(args.output_dir)

    dbscan_anomaly_detector.register_detection_callback(
                    json_anomaly_registry.clustering_detection)
    informer_runner.register_detection_callback(
                    json_anomaly_registry.transformer_detection)
    
    data_loader.force_sso_login()

    aio.run(wait_for_states(['RUNNING'],
                    dt.timedelta(milliseconds=1)))

#     aio.run(wait_for_state('RUNNING',
#                         dt.timedelta(minutes=5)))
#     
#     # Process a number of elements equal to the ReducedDataBuffer 
#     # size minus one so the transformer-based detection can 
#     # begin detection on the first polled sample
# 
#     time_start = t.monotonic()
# 
#     with console.status('Prefilling buffer...', spinner='flip'):
# 
#         buffer_prefill_chunk =\
#             data_loader.get_prefill_chunk(reduced_data_buffer_size - 1)
# 
#         for element in np.vsplit(buffer_prefill_chunk,
#                                     reduced_data_buffer_size - 1):
# 
#             timestamp = element.index[0]
# 
#             data = element.to_numpy().squeeze()
# 
#             try:
#                 dbscan_anomaly_detector.process(timestamp, data)
# 
#                 output_slice =\
#                     median_std_reducer.reduce_numpy(tpu_labels,
#                                                         timestamp,
#                                                         data)
# 
#                 output_slice = fix_for_2023_deployment(output_slice)
# 
#                 reduced_data_buffer.push(output_slice)
#                 
#             except NonCriticalPredictionException:
#                 break
# 
#     prefill_duration = t.monotonic() - time_start
# 
#     logger.info(f'Buffer prefill took {prefill_duration:.3f} s')

    def processing_func(queue):

        processed_element_count = 0

        with console.status('Running anomaly detection...',
                                            spinner='flip',
                                            speed=0.5,
                                            refresh_per_second=5):
            while True:

                element = queue.get()

                if element is None:
                    break

                timestamp = element.index[0]

                data = element.to_numpy().squeeze()

                try:
                    dbscan_anomaly_detector.process(timestamp, data)

                    output_slice =\
                        median_std_reducer.reduce_numpy(tpu_labels,
                                                            timestamp,
                                                            data)
                    
                    # output_slice = fix_for_2023_deployment(output_slice)
                    
                    reduced_data_buffer.push(output_slice)
                    
                except NonCriticalPredictionException:
                    break

                if (processed_element_count > 0) and\
                            ~(processed_element_count %\
                                args.anomaly_log_dump_interval):
                    time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    anomaly_log_name = f'anomaly_log_{time_now_string}'
                    json_anomaly_registry.dump(anomaly_log_name)

            time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            anomaly_log_name = f'anomaly_log_{time_now_string}'
            json_anomaly_registry.dump(anomaly_log_name)

    queue = mp.Queue()

    close_event = mp.Event()

    data_loader_proc = mp.Process(target=data_loader.poll,
                                    args=(queue, close_event,))
    data_loader_proc.start()

    try:
        processing_func(queue)
    except KeyboardInterrupt:
        logger.info('Received keyboard interrupt')

        time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        anomaly_log_name = f'anomaly_log_{time_now_string}'
        json_anomaly_registry.dump(anomaly_log_name)
        
        close_event.set()
        data_loader_proc.join()
        raise

    except Exception as e:
        logger.error(f'Caught exception: {e}')
        close_event.set()
        raise
    else:
        data_loader_proc.join()