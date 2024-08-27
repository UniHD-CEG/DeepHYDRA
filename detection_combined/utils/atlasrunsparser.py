#!/usr/bin/env python3

import datetime as dt
from enum import Enum
from html.parser import HTMLParser

import pandas as pd

class AtlasRunsParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)

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
                
                self._run_info['start'] = dt.datetime.strptime('2018 ' + data, '%Y %a %b %d, %H:%M %Z')
                
            elif self._run_info_data_type is self.RunInfoDataType.run_end:
                
                assert(self._run_info['end'] is None)
                
                self._run_info['end'] = dt.datetime.strptime('2018 ' + data, '%Y %a %b %d, %H:%M %Z')
                
                duration_dt = self._run_info['end'] - self._run_info['start']
                
                self._run_info['duration'] = duration_dt.total_seconds()
                
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


