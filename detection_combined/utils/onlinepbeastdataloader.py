#!/usr/bin/env python3

import os
import time as t
import datetime as dt
import asyncio
import logging

import numpy as np
import pandas as pd
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
                        'https://vm-atlas-tdaq-cc.cern.ch/tbed/pbeast/api/') -> None:

        self._data_channel = data_channel
        self._polling_interval = polling_interval
        self._delay = delay
        self._window_length = window_length
        self._pbeast_server = pbeast_server

        os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

        self._beauty_instance = Beauty(server=self._pbeast_server)

        self._column_names = None
        self._initialized = False

        self._logger = logging.getLogger(__name__)


    def init(self) -> pd.DataFrame:
        data_channel_vars = _data_channel_vars_dict['DCMRate']

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
            self._logger.error('Could not read DCM rate data from PBEAST')
            raise

        self._logger.info('Successfully retrieved PBEAST data')

        for count in range(1, len(dcm_rates_all_list)):
            dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

        dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

        self._column_names = dcm_rates_all_pd.columns
        self._initialized = True


    async def poll(self) -> pd.DataFrame:

        try:

            data_channel_vars = _data_channel_vars_dict['DCMRate']

            while True:

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
                    self._logger.error('Could not read DCM rate data from PBEAST')
                    break

                self._logger.debug('Successfully retrieved PBEAST data')

                for count in range(1, len(dcm_rates_all_list)):
                    dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

                dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

                yield dcm_rates_all_pd.fillna(nan_fill_value)

                request_duration = t.monotonic() - time_start

                if request_duration >= self._polling_interval.total_seconds():

                    error_string = 'Request processing time '\
                                        'exceeded polling interval.'\
                                        f'Request processing time: {request_duration} s\t'\
                                        'Polling interval: '\
                                        f'{self._polling_interval.total_seconds()} s'

                    self._logger.error(error_string)
                    raise RuntimeError(error_string)

                delay_period = self._polling_interval.total_seconds() - request_duration

                await asyncio.sleep(delay_period)

            yield None

        except asyncio.CancelledError:
            self._logger.info('Received stop request. Exiting')


    def get_column_names(self) -> list:
        if not self._initialized:
            error_string = 'Not initialized'
            self._logger.error('Could not read DCM rate data from PBEAST')
            raise RuntimeError(error_string)

        return list(self._column_names)