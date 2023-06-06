#!/usr/bin/env python3

import os
import logging

import numpy as np
import pandas as pd
from beauty import Beauty

from .atlasrunsparser import AtlasRunsParser
from .variables import nan_fill_value

_data_channel_vars_dict = {'L1Rate': ['ATLAS', 'DCM', 'L1Rate', 'DF_IS:.*.DCM.*.info']}

nan_fill_value = np.finfo(np.float32).min


class OfflinePBeastDataLoader():

    def __init__(self, data_channel: str,
                        runs_summary_filename: str) -> None:

        self._data_channel = data_channel

        os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

        # self._beauty_instance = Beauty()
        self._beauty_instance = Beauty(server='https://vm-atlas-tdaq-cc.cern.ch/tbed/pbeast/api/')

        with open(runs_summary_filename) as file:
            html_string = file.read()

        atlas_runs_parser = AtlasRunsParser()

        atlas_runs_parser.feed(html_string)
        atlas_runs_parser.close()

        self._runs_df = atlas_runs_parser.runs

        self._run_numbers_all = list(self._runs_df.index.values)

        self._logger = logging.getLogger(__name__)


    def get_run_numbers(self) -> list:
        return self._run_numbers_all


    def __getitem__(self, run_number: int) -> pd.DataFrame:

        if run_number not in self._run_numbers_all:
            raise RuntimeError('No data available for selected run number')

        run_data = self._runs_df.loc[run_number]
        
        time_start = run_data['start']
        time_end = run_data['end']
        duration = run_data['duration']

        self._logger.info(f'Reading data for run: {run_number}')
        self._logger.info(f'Start time: {time_start}')
        self._logger.info(f'End time: {time_end}')
        self._logger.info(f'Duration: {int(duration):d} s')

        data_channel_vars = _data_channel_vars_dict['L1Rate']

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
            self._logger.error('Could not read DCM rate data '
                                f'for run {run_number} from PBEAST')
            raise

        for count in range(1, len(dcm_rates_all_list)):
            dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

        dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

        return dcm_rates_all_pd.fillna(nan_fill_value)


    def __len__(self) -> int:
        return len(self._run_numbers_all) 