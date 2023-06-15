#!/usr/bin/env python3

import os
import time as t
import datetime as dt
import multiprocessing as mp
import logging

import numpy as np
import pandas as pd
from rich import print
from beauty import Beauty

from .variables import nan_fill_value

_data_channel_vars_dict = {'DCMRate': ['ATLAS', 'DCM', 'L1Rate', 'DF_IS:.*.DCM.*.info']}

_data_refresh_rates_seconds_dict = {'DCMRate': 5}


_backup_tpu_name_reference_timestamp = dt.datetime(2023, 6, 9, 0, 0, 0)


class OnlinePBeastDataLoader():

    def __init__(self,
                    data_channel: str,
                    polling_interval: dt.timedelta = dt.timedelta(seconds=10),
                    delay: dt.timedelta = dt.timedelta(seconds=30),
                    window_length: dt.timedelta = dt.timedelta(seconds=10),
                    pbeast_server: str = 'https://atlasop.cern.ch',
                    data_unavail_handling: str = 'raise_exception') -> None:

        self._data_channel = data_channel
        self._polling_interval = polling_interval
        self._delay = delay
        self._window_length = window_length
        self._pbeast_server = pbeast_server
        
        if data_unavail_handling not in ['raise_exception',
                                                'return_zeros']:
            
            raise ValueError('Data unavailability handling type '
                                f'{data_unavail_handling} is unknown')
        
        self._data_unavail_handling = data_unavail_handling

        os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

        self._beauty_instance = Beauty(server=self._pbeast_server)

        self._column_names = None
        self._timestamp_last = None
        self._initialized = False

        self._logger = logging.getLogger(__name__)


    def _get_idx_closest_consecutive_timestamp(self,
                                                timestamps: pd.DatetimeIndex):
        
        cand_timestamps = [timestamp for timestamp in timestamps\
                                if timestamp > self._timestamp_last]
        
        if len(cand_timestamps) == 0:
            error_string = 'No consecutive timestamp found in input'
            self._logger.error(error_string)
            raise RuntimeError(error_string)

        return np.argmin(cand_timestamps)


    def init(self) -> pd.DataFrame:
        data_channel_vars = _data_channel_vars_dict[self._data_channel]

        requested_period_end = dt.datetime.now() - self._delay
        requested_period_start = requested_period_end - self._window_length

        self._logger.info('Initializing')

        self._logger.debug(f'Requesting PBEAST data from '
                            f'{requested_period_start} to {requested_period_end}')
        self._logger.debug(f'Request vars: {data_channel_vars[0]}, {data_channel_vars[1]}, '
                                            f'{data_channel_vars[2]}, {data_channel_vars[3]}')

        dcm_rates_all_list = []

        try:
            dcm_rates_all_list = self._beauty_instance.timeseries(requested_period_start,
                                                                    requested_period_end,
                                                                    data_channel_vars[0],
                                                                    data_channel_vars[1],
                                                                    data_channel_vars[2],
                                                                    data_channel_vars[3],
                                                                    regex=True,
                                                                    all_publications=True)

        except RuntimeError as runtime_error:
            if self._data_unavail_handling == 'return_zeros':
                self._logger.warning(f'PBEAST data retrieval to obtain channel name '
                                        'reference failed, trying backup request time. '
                                        'The obtained channel names might not line up '
                                        'with the current channels, which might '
                                        'will cause the detection to fail.')

            else:
                self._logger.error(f'Could not read {self._data_channel} data from PBEAST')
                raise


        if len(dcm_rates_all_list) == 0:
            try:
                dcm_rates_all_list = self._beauty_instance.timeseries(_backup_tpu_name_reference_timestamp,
                                                                        _backup_tpu_name_reference_timestamp,
                                                                        data_channel_vars[0],
                                                                        data_channel_vars[1],
                                                                        data_channel_vars[2],
                                                                        data_channel_vars[3],
                                                                        regex=True,
                                                                        all_publications=True)

            except RuntimeError as runtime_error:
                self._logger.error('Backup channel name retrieval failed. '
                                        'PBEAST service might be unavailable')
                raise

        # for count in range(1, len(dcm_rates_all_list)):
        #     dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

        for count in range(1, len(dcm_rates_all_list)):
            dcm_rates_all_list[count].index = dcm_rates_all_list[0].index

        dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

        self._column_names = dcm_rates_all_pd.columns
        self._initialized = True

        self._logger.info('Initialization successful')


    def get_prefill_chunk(self,
                            size: int) -> pd.DataFrame:

        data_channel_vars = _data_channel_vars_dict[self._data_channel]

        requested_period_end = dt.datetime.now() - self._delay - self._window_length
        requested_period_start = requested_period_end - size*self._polling_interval

        self._logger.info('Requesting prefill chunk')

        self._logger.debug(f'Requesting PBEAST data from '
                            f'{requested_period_start} to {requested_period_end}')
        self._logger.debug(f'Request vars: {data_channel_vars[0]}, {data_channel_vars[1]}, '
                                            f'{data_channel_vars[2]}, {data_channel_vars[3]}')

        dcm_rates_all_list = []

        try:
            dcm_rates_all_list = self._beauty_instance.timeseries(requested_period_start,
                                                                    requested_period_end,
                                                                    data_channel_vars[0],
                                                                    data_channel_vars[1],
                                                                    data_channel_vars[2],
                                                                    data_channel_vars[3],
                                                                    regex=True,
                                                                    all_publications=True)

        except RuntimeError as runtime_error:
            if self._data_unavail_handling != 'return_zeros':
                self._logger.error(f'Could not read {self._data_channel} data from PBEAST')
                raise

        if len(dcm_rates_all_list) != 0:
            self._logger.debug('Successfully retrieved PBEAST data')

            # for count in range(1, len(dcm_rates_all_list)):
            #     dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

            # timestamps = [element.index[-1].to_numpy().astype(np.int64) for element in dcm_rates_all_list]

            # mean_val = np.datetime64(int(np.mean(timestamps)), 'ns')
            # min_val = np.datetime64(int(np.min(timestamps)), 'ns')
            # max_val = np.datetime64(int(np.max(timestamps)), 'ns')
            # std = np.std(timestamps)

            # print(f'{mean_val}\t{min_val}\t{max_val}\t{std:.3f}')

            for count in range(1, len(dcm_rates_all_list)):
                # TODO: Add handling of unequal lengths
                # of returned Series
                dcm_rates_all_list[count].index = dcm_rates_all_list[0].index

            dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

            if len(dcm_rates_all_pd) < size:

                self._logger.info('Got smaller prefill chunk than '
                                    'requested. Pre-padding with zeros')

                print(dcm_rates_all_pd.shape)
                
                pad_length = size - len(dcm_rates_all_pd)

                zeros = np.zeros((pad_length), dcm_rates_all_pd.shape[-1])

                refresh_rate =\
                    _data_refresh_rates_seconds_dict['DCMRate']

                start_datetime_zeros =\
                    dcm_rates_all_pd.index[0] -\
                        pd.Timedelta(pad_length*refresh_rate, unit='S')

                index_zeros = pd.Series(pd.date_range(
                                            start=start_datetime_zeros,
                                            end=dcm_rates_all_pd.index[0],
                                            freq=f'{refresh_rate}S',
                                            inclusive='left'))
                
                index_zeros = pd.DatetimeIndex(index_zeros)

                dcm_rates_all_pd = pd.concat((pd.DataFrame(zeros,
                                                index=index_zeros,
                                                columns=dcm_rates_all_pd.columns),
                                                dcm_rates_all_pd))

            else:
                dcm_rates_all_pd = dcm_rates_all_pd.iloc[-size:, :]

            dcm_rates_all_pd = dcm_rates_all_pd.fillna(nan_fill_value)

        else:
            self._logger.warning('Prefill chunk data retrieval failed '
                                    'with chosen handling behavior '
                                    f'{self._data_unavail_handling}.'
                                    'Returning all-zero chunk.')
            
            index = pd.date_range(requested_period_start,
                                        requested_period_end,
                                        size)
            
            print(index)

            zeros = np.zeros((size, len(self._column_names)))

            print(zeros.shape)

            dcm_rates_all_pd = pd.DataFrame(zeros, index)

            print(dcm_rates_all_pd.shape)
            
        self._timestamp_last = dcm_rates_all_pd.index[-1]

        print(requested_period_start)
        print(dcm_rates_all_pd.index[0])

        print(requested_period_end)
        print(self._timestamp_last)

        self._logger.info('Prefill chunk processing successful')

        return dcm_rates_all_pd
    

    def poll(self,
                queue: mp.Queue,
                close_event: mp.Event) -> pd.DataFrame:

        data_channel_vars = _data_channel_vars_dict[self._data_channel]

        while not close_event.is_set():

            time_start = t.monotonic()

            requested_period_end = dt.datetime.now() - self._delay
            requested_period_start = requested_period_end - self._window_length

            self._logger.debug(f'Requesting PBEAST data from '
                                f'{requested_period_start} to {requested_period_end}')
            self._logger.debug(f'Request vars: {data_channel_vars[0]}, {data_channel_vars[1]}, '
                                                f'{data_channel_vars[2]}, {data_channel_vars[3]}')

            try:
                dcm_rates_all_list = self._beauty_instance.timeseries(requested_period_start,
                                                                        requested_period_end,
                                                                        data_channel_vars[0],
                                                                        data_channel_vars[1],
                                                                        data_channel_vars[2],
                                                                        data_channel_vars[3],
                                                                        regex=True,
                                                                        all_publications=True)

            except RuntimeError as runtime_error:
                self._logger.error(f'Could not read {self._data_channel} data from PBEAST')
                break

            self._logger.debug('Successfully retrieved PBEAST data')

            # for count in range(1, len(dcm_rates_all_list)):
            #     dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

            for count in range(1, len(dcm_rates_all_list)):
                dcm_rates_all_list[count].index = dcm_rates_all_list[0].index

            dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

            dcm_rates_all_pd = dcm_rates_all_pd.fillna(nan_fill_value)

            target_idx = self._get_idx_closest_consecutive_timestamp(dcm_rates_all_pd.index)

            dcm_rates_all_pd = dcm_rates_all_pd.iloc[[target_idx], :]

            timestamp_delta = dcm_rates_all_pd.index[0] - self._timestamp_last

            self._logger.debug(f'Current timestamp: {dcm_rates_all_pd.index[0]}')
            self._logger.debug(f'Last timestamp: {self._timestamp_last}')
            self._logger.debug(f'Current timestamp delta: {timestamp_delta}')

            self._timestamp_last = dcm_rates_all_pd.index[0]
            
            queue.put(dcm_rates_all_pd)

            request_duration = t.monotonic() - time_start

            if request_duration >= self._polling_interval.total_seconds():

                queue.put(None)

                error_string = 'Request processing time '\
                                    'exceeded polling interval. '\
                                    f'Request processing time: {request_duration:.3f} s\t'\
                                    'Polling interval: '\
                                    f'{self._polling_interval.total_seconds()} s'

                self._logger.error(error_string)
                raise RuntimeError(error_string)

            delay_period = self._polling_interval.total_seconds() - request_duration

            t.sleep(delay_period)

        queue.put(None)


    def get_column_names(self) -> list:
        if not self._initialized:
            error_string = 'Not initialized'
            self._logger.error(error_string)
            raise RuntimeError(error_string)

        return list(self._column_names)