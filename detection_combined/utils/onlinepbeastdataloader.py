#!/usr/bin/env python3

import os
import time as t
import datetime as dt
import multiprocessing as mp
import asyncio
import logging

import numpy as np
import pandas as pd
from rich import print
from beauty import Beauty


from .variables import nan_fill_value

_data_channel_vars_dict = {'DCMRate': ['ATLAS', 'DCM', 'L1Rate', 'DF_IS:.*.DCM.*.info']}


class OnlinePBeastDataLoader():

    def __init__(self,
                    data_channel: str,
                    polling_interval: dt.datetime = dt.timedelta(seconds=10),
                    delay: dt.timedelta = dt.timedelta(seconds=30),
                    window_length: dt.timedelta = dt.timedelta(seconds=10),
                    pbeast_server: str =\
                        'https://atlasop.cern.ch') -> None:

        self._data_channel = data_channel
        self._polling_interval = polling_interval
        self._delay = delay
        self._window_length = window_length
        self._pbeast_server = pbeast_server

        os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

        self._beauty_instance = Beauty(server=self._pbeast_server)

        self._column_names = None
        self._timestamp_last = None
        self._initialized = False

        self._logger = logging.getLogger(__name__)


    def get_idx_closest_consecutive_timestamp(self,
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
            raise

        self._logger.info('Successfully retrieved PBEAST data')

        for count in range(1, len(dcm_rates_all_list)):
            dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

        dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

        self._column_names = dcm_rates_all_pd.columns
        self._initialized = True


    def get_prefill_chunk(self,
                            size: int) -> pd.DataFrame:

        data_channel_vars = _data_channel_vars_dict[self._data_channel]

        requested_period_end = dt.datetime.now() - self._delay - self._window_length
        requested_period_start = requested_period_end - size*self._polling_interval

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
            raise

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
            dcm_rates_all_list[count].index = dcm_rates_all_list[0].index

        dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

        dcm_rates_all_pd = dcm_rates_all_pd.fillna(nan_fill_value)

        dcm_rates_all_pd = dcm_rates_all_pd.iloc[-size:, :]

        self._timestamp_last = dcm_rates_all_pd.index[-1]

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

            target_idx = self.get_idx_closest_consecutive_timestamp(dcm_rates_all_pd.index)

            dcm_rates_all_pd = dcm_rates_all_pd.iloc[[target_idx], :]

            timestamp_delta = dcm_rates_all_pd.index[0] - self._timestamp_last

            print(f'Current timestamp: {dcm_rates_all_pd.index[0]}')
            print(f'Last timestamp: {self._timestamp_last}')

            self._logger.debug(f'Current timestamp delta: {timestamp_delta}')

            self._timestamp_last = dcm_rates_all_pd.index[0]
            
            queue.put(dcm_rates_all_pd)

            request_duration = t.monotonic() - time_start

            if request_duration >= self._polling_interval.total_seconds():

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
            self._logger.error('Could not read DCM rate data from PBEAST')
            raise RuntimeError(error_string)

        return list(self._column_names)