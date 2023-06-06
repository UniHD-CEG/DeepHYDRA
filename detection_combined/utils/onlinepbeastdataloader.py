#!/usr/bin/env python3

import os
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

        self._logger = logging.getLogger(__name__)


    async def poll(self) -> pd.DataFrame:

        try:

            data_channel_vars = _data_channel_vars_dict['DCMRate']

            while True:

                time_end = dt.now() - self._delay
                time_start = time_end - self._window_length

                self._logger.info(f'Requesting PBEAST data from {time_start} to {time_end}')
                self._logger.info(f'Request vars: {data_channel_vars[0]}, {data_channel_vars[1]}, '
                                                    f'{data_channel_vars[2]}, {data_channel_vars[3]}')

                try:
                    dcm_rates_all_list = self._beauty_instance.timeseries(time_start,
                                                                            time_end,
                                                                            data_channel_vars[0],
                                                                            data_channel_vars[1],
                                                                            data_channel_vars[2],
                                                                            data_channel_vars[3],
                                                                            regex=True,
                                                                            all_publications=True)

                except RuntimeError as runtime_error:
                    self._logger.error('Could not read DCM rate data from PBEAST')
                    break

                self._logger.info('Successfully retrieved PBEAST data')

                for count in range(1, len(dcm_rates_all_list)):
                    dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

                dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

                yield dcm_rates_all_pd.fillna(nan_fill_value)

                await asyncio.sleep(self._polling_interval.total_seconds())

            yield None

        except asyncio.CancelledError:
            self._logger.info('Received stop request. Exiting')