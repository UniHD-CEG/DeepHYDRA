#!/usr/bin/env python3

import argparse
import sys
import datetime as dt
import multiprocessing as mp
import asyncio
import json
import logging


import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.append('../../')

from clustering.dbscananomalydetector import DBScanAnomalyDetector
from reduction.medianstdreducer import MedianStdReducer
from transformer_based_detection.informers.informerrunner import InformerRunner
from utils.onlinepbeastdataloader import OnlinePBeastDataLoader
from utils.anomalyregistry import JSONAnomalyRegistry
from utils.reduceddatabuffer import ReducedDataBuffer
from utils.exceptions import NonCriticalPredictionException


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='T-DBSCAN/Informer Offline HLT Anomaly Detection')

    parser.add_argument('--model', type=str, choices=['Informer-MSE', 'Informer-SMSE'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint-dir', type=str, default='../../../transformer_based_detection')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--spot-based-detection', action='store_true', default=False)

    parser.add_argument('--dbscan-eps', type=float, default=3)
    parser.add_argument('--dbscan-min-samples', type=int, default=4)
    parser.add_argument('--dbscan-duration-threshold', type=int, default=4)

    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--log-level', type=str, default='info')
    parser.add_argument('--log-dir', type=str, default='./log/')
    parser.add_argument('--anomaly-log-dump-interval', type=int, default=720)

    args = parser.parse_args()

    time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_model_name = args.model.lower().replace('-', '_')

    log_filename = f'{args.log_dir}/strada_{log_model_name}'\
                            f'_log_{time_now_string}.log'

    logging_format = '[%(asctime)s] %(levelname)s: %(name)s: %(message)s'

    logger = logging.getLogger(__name__)

    logging.getLogger().addFilter(lambda record: 'running accelerated version on CPU' not in record.msg)

    logging.basicConfig(filename=log_filename,
                                    filemode='w',
                                    level=args.log_level.upper(),
                                    format=logging_format,
                                    datefmt='%Y-%m-%d %H:%M:%S')
    
    data_loader = OnlinePBeastDataLoader('DCMRate',
                                            polling_interval=dt.timedelta(seconds=5))

    data_loader.init()

    tpu_labels = data_loader.get_column_names()

    median_std_reducer = MedianStdReducer()

    loss_type = args.model.split('-')[-1].lower()
    
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
    
    # Process a number of elements equal to the ReducedDataBuffer 
    # size minus one so the transformer-based detection can 
    # begin detection on the first polled sample
    
    buffer_prefill_chunk =\
        data_loader.get_prefill_chunk(reduced_data_buffer_size - 1)

    for element in np.vsplit(buffer_prefill_chunk,
                                reduced_data_buffer_size - 1):

        timestamp = element.index[0]

        data = element.to_numpy().squeeze()

        try:
            dbscan_anomaly_detector.process(timestamp, data)

            output_slice =\
                median_std_reducer.reduce_numpy(tpu_labels,
                                                    timestamp,
                                                    data)
            reduced_data_buffer.push(output_slice)
            
        except NonCriticalPredictionException:
            break

    def processing_func(queue):

        processed_element_count = 0

        for element in queue:

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

    async def async_main():

        loop = asyncio.get_event_loop()
        queue = mp.Queue()

        data_loader_task = loop.run_in_executor(None, data_loader.poll)

        processing_proc = mp.Process(target=processing_func, args=(queue,))
        processing_proc.start()

        async for item in data_loader_task:
            queue.put(item)

        processing_proc.join()
        

    asyncio.run(async_main())

#     @asyncio.coroutine
#     def data_loader_func():
#         yield from data_loader.poll()
# 
#     try:
# 
#         data_loader_proc = mp.Process(target=asyncio.run, args=(data_loader_func(),))
#         processing_proc = mp.Process(target=processing_func)
# 
#         data_loader_proc.start()
#         processing_proc.start()
# 
#         data_loader_proc.join()
#         processing_proc.join()
# 
#     except KeyboardInterrupt:
#         data_loader_proc.terminate()
#         processing_proc.terminate()