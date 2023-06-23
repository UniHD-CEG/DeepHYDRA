#!/usr/bin/env python3
import re
import argparse
import datetime as dt
from enum import Enum
from collections import defaultdict
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

failure_source_dict =  {'DCM':              0b00000000001000000,
                        'HLT':              0b00000000010000000,
                        'HLTRC':            0b00000000100000000,
                        'HLTMPPU':          0b00000001000000000,
                        'HLT_DB_Gen':       0b00000010000000000,
                        'DF_IS':            0b00000100000000000,
                        'DF_Histogramming': 0b00001000000000000,
                        'DefRDB':           0b00010000000000000,
                        'NodeCoralProxy':   0b00100000000000000,
                        'RackCoralProxy':   0b01000000000000000,
                        'hardware_glitch':  0b10000000000000000}


class AtlasRunsParser(HTMLParser):

    def __init__(self, variant: str):
        HTMLParser.__init__(self)

        self._variant = variant

        self._runs_df = pd.DataFrame(columns = ['run', 'start', 'end', 'duration'])

        self._run_info_data_type = self.RunInfoDataType(self.RunInfoDataType.dont_care)
        self._run_info = {'run': None, 'start': None, 'end': None, 'duration': None}

    def handle_data(self, data):
        
        if self._run_info_data_type is self.RunInfoDataType.dont_care:
        
            if data == 'Run ':
                self._run_info_data_type = self.RunInfoDataType.run_number
            elif data == 'Start':
                self._run_info_data_type = self.RunInfoDataType.run_start
            elif data == 'End':
                self._run_info_data_type = self.RunInfoDataType.run_end
                
        else:
            
            if self._run_info_data_type is self.RunInfoDataType.run_number:
                
                assert(self._run_info['run'] is None)
                
                self._run_info['run'] = int(data)
                
            elif self._run_info_data_type is self.RunInfoDataType.run_start:
                
                assert(self._run_info['start'] is None)
                
                self._run_info['start'] = dt.datetime.strptime(f'{self._variant} ' + data, '%Y %a %b %d, %H:%M %Z')
                
            elif self._run_info_data_type is self.RunInfoDataType.run_end:
                
                assert(self._run_info['end'] is None)
                
                self._run_info['end'] = dt.datetime.strptime(f'{self._variant} ' + data, '%Y %a %b %d, %H:%M %Z')
                
                duration_dt = self._run_info['end'] - self._run_info['start']
                
                self._run_info['duration'] = int(duration_dt.total_seconds())
                
                self._runs_df = self._runs_df.append(self._run_info, ignore_index=True)
                
                self._run_info = {'run': None, 'start': None, 'end': None, 'duration': None}
        
            else:
                raise RuntimeError('AtlasRunsParser entered unexpected state')
                        
            self._run_info_data_type = self.RunInfoDataType.dont_care

    @property
    def runs(self):
        return self._runs_df.iloc[::-1].set_index('run')
        
    class RunInfoDataType(Enum):
        dont_care = 0
        run_number = 1
        run_start = 2
        run_end = 3


if __name__ == '__main__':

    pd.set_option('display.max_rows', 2048)

    parser = argparse.ArgumentParser()

    parser.add_argument('--variant', type=str)
    parser.add_argument('--input-filename', type=str)
    parser.add_argument('--output-filename', type=str)
    parser.add_argument('--run-summary-file', type=str,
                            default='../../atlas-data-summary-runs-2022.html')
    
    args = parser.parse_args()

    atlas_runs_2022 = None

    with open(args.run_summary_file) as file:
        html_string = file.read()

        atlas_runs_parser = AtlasRunsParser(args.variant)

        atlas_runs_parser.feed(html_string)
        atlas_runs_parser.close()

        atlas_runs_2022 = atlas_runs_parser.runs

    tpu_failure_log = pd.read_csv(args.input_filename)

    tpu_failure_log['time'] = pd.to_datetime(
                                    tpu_failure_log['time'],
                                    format='%H:%M:%S %b %d %Y')

    tpu_failure_log['failure_source'] =\
            tpu_failure_log['failure_source'].map(failure_source_dict)

    tpu_failure_log.set_index(['time', 'tpu'],
                                    inplace=True)

    tpu_failure_log = tpu_failure_log.iloc[::-1]

    tpu_failure_log =\
        tpu_failure_log.groupby(level=[0, 1]).sum()

    tpu_failure_log = tpu_failure_log.iloc[::-1]

    tpu_failure_log.reset_index(level='tpu',
                                    inplace=True)

    tpu_failure_log['start'] = tpu_failure_log.index

    run_numbers = pd.Series(pd.NA, tpu_failure_log.index)
    failure_ends = pd.Series(pd.NA, tpu_failure_log.index)

    failure_count_per_run = defaultdict(int)

    for run in atlas_runs_2022.itertuples():

        failure_count_per_run[run.Index] = 0

        for count in range(len(tpu_failure_log)):

            failure_start =\
                tpu_failure_log.iloc[count]['start']

            if failure_start >= run.start and\
                            failure_start < run.end:
                run_numbers.iloc[count] = run.Index
                failure_ends.iloc[count] = run.end

                failure_count_per_run[run.Index] += 1

    for run, failure_count in failure_count_per_run.items():
        print(f'Run {run}: {failure_count} failures')

    tpu_failure_log['run'] = run_numbers
    tpu_failure_log['end'] = failure_ends

    tpu_failure_log.dropna(how='any', inplace=True)

    tpu_failure_log.set_index(['run', 'tpu'], inplace=True)

    tpu_failure_log.to_hdf(args.output_filename,
                                key='tpu_failure_log_preprocessed',
                                mode='w')


