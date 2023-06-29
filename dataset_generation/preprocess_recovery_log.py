#!/usr/bin/env python3
import re
import argparse
import datetime as dt
from enum import Enum
from html.parser import HTMLParser
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    warnings.simplefilter('ignore',
                            category=FutureWarning)

    pd.set_option('display.max_rows', 2048)

    parser = argparse.ArgumentParser()

    parser.add_argument('--variant', type=str)
    parser.add_argument('--input-filename', type=str)
    parser.add_argument('--output-filename', type=str)
    parser.add_argument('--run-summary-file', type=str,
                            default='../../atlas-data-summary-runs-2022.html')
    
    args = parser.parse_args()

    atlas_runs = None

    with open(args.run_summary_file) as file:
        html_string = file.read()

        atlas_runs_parser = AtlasRunsParser(args.variant)

        atlas_runs_parser.feed(html_string)
        atlas_runs_parser.close()

        atlas_runs = atlas_runs_parser.runs

    recoveries_hierarchical = pd.read_hdf(args.input_filename,
                                        key='recoveries_hierarchical')

    recoveries_hierarchical['time'] = pd.to_datetime(
                                            recoveries_hierarchical['time'],
                                            format='%H:%M:%S %b %d %Y')
    
    recovery_failures = pd.DataFrame(columns=['run', 'source', 'start', 'end'])

    failure_count = 0

    for count, run in enumerate(atlas_runs.itertuples()):

        print(f'Run {run.Index}')

        for label, data in recoveries_hierarchical.groupby(level=0):

            timestamps = data['time'].values
            types = data['type'].values

            in_run = np.logical_and(timestamps >= run.start,
                                        timestamps <= run.end)

            timestamps_in_run = timestamps[in_run]
            types_in_run = types[in_run]

            if 'F' in types_in_run:
                for timestamp, mode in zip(timestamps_in_run, types_in_run):
                    print(f'{label}')

                    # This only works for logs where there are no
                    # successful recoveries after failed recoveries
                    # in the runs contained in the run summary file.
                    # If in the future such cases are included in the
                    # processed logs, one will have to add functionality
                    # to check if a successful recovery with a matching
                    # label followed the failed recovery. In this case,
                    # the end of the anomaly would be the timestamp of
                    # the first successful matching recovery after the
                    # failed recovery, not the end of the run.

                    print(f'Time start: {timestamp}\ttime end: {run.end}')

                    if mode == 'F':
                        recovery_failures.loc[failure_count] =\
                                                    {'run': run.Index,
                                                        'source': label,
                                                        'start': timestamp,
                                                        'end': run.end}
                        
                        failure_count += 1

    # recovery_failures.to_hdf(args.output_filename,
    #                             key='recovery_log_preprocessed',
    #                             mode='w')

