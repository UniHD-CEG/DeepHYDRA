import math
import re
import multiprocessing as mp
import ipaddress
import logging
from collections import deque
from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr


plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

_rack_numbers_expected_2023 =   [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [10, 11, 12, 13, 17, 18, 19],
                                    [20, 21, 22, 23, 24, 25, 26],
                                    [44, 45, 46, 47, 48, 49],
                                    [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                                    [60, 61, 62, 64, 65, 66, 67, 68, 69],
                                    [70, 71, 72, 73, 74, 75, 76, 77, 79],
                                    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                                    [90, 91, 92, 93, 94, 95]]

_rack_status =  [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]


def get_anomalous_runs(x):
    '''
    Find runs of consecutive items in an array.
    As published in https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    '''

    # Ensure array

    x = np.asanyarray(x)

    if x.ndim != 1:
        raise ValueError('Only 1D arrays supported')

    n = x.shape[0]

    # Handle empty array

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:

        # Find run starts

        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True

        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # Find run values
        run_values = x[loc_run_start]

        # Find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        run_starts = np.compress(run_values, run_starts)
        run_lengths = np.compress(run_values, run_lengths)

        run_ends = run_starts + run_lengths

        return run_starts, run_ends


class GradioServer():
    def __init__(self,
                    clustering_queue: mp.Queue,
                    time_series_queue: mp.Queue,
                    log_queue: mp.Queue,
                    address: str = 'localhost',
                    auth_data = None) -> None:

        self._clustering_queue = clustering_queue
        self._time_series_queue = time_series_queue
        self._log_queue = log_queue

        self._logger = logging.getLogger(__name__)

        try:
            self._address = str(ipaddress.ip_address(address))
        except ValueError:
            error_string =\
                f'{address} is not a valid IP address'
        
            self._logger.error(error_string)
            raise ValueError(error_string)

        if auth_data is not None:
            if not isinstance(auth_data, Iterable):
                error_string = 'auth_data must be an Iterable'
            
                self._logger.error(error_string)
                raise ValueError(error_string)

            username = auth_data[0]
            password = auth_data[1]
            
            self._validate_username(username)
            self._validate_password(password)

        self._auth_data = auth_data

        self._time_series_buffer = deque([], maxlen=64)
        self._log_buffer = deque([], maxlen=25)

        self._time_series_plot = None
        self._time_series_plot_mutex = mp.Lock()

        self._dashboard = gr.Blocks(title='STRADA Dashboard',
                                        analytics_enabled=False)

        with self._dashboard:
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        gr.Textbox('Transformer-Based Detection',
                                        label='', interactive=False)
                    with gr.Row():
                        self._plot =\
                            gr.Plot()
                with gr.Column(scale=1):
                    with gr.Row():
                        gr.Textbox('T-DBSCAN-Based Detection',
                                        label='', interactive=False)
                    with gr.Row():
                        self._cluster_status =\
                            gr.HTML(label='Clustering-Based Detection')
            with gr.Row():
                self._log_output = gr.Textbox(label='Log Output',
                                                        lines=25,
                                                        interactive=False)

            self._dashboard.load(self._get_rack_grid, None, 
                                    self._cluster_status, every=5)
            
            self._dashboard.load(self._get_time_series_plot, None,
                                                self._plot, every=5)

            self._dashboard.load(self._generate_time_series_plot,
                                                None, None, every=5)

            self._dashboard.load(self._get_log_string, None,
                                    self._log_output, every=1)


    def _validate_username(self,
                            username: str):
        if not isinstance(username, str):
            error_string = 'username must be a string'
            
            self._logger.error(error_string)
            raise ValueError(error_string)
        
        pattern = r'^[a-zA-Z0-9_-]+$'
        if not re.match(pattern, username):

            error_string = 'Username contains illegal characters. '\
                            'Only alphanumeric characters, "_" and "-" are allowed.'
            
            self._logger.error(error_string)
            raise ValueError(error_string)


    def _validate_password(self,
                        password: str):
        
        # Although they are functionally identical right now,
        # this is a distinct function from validate_username()
        # in order to facilitate the usage of different
        # character whitelists for usernames and passwords in
        # the future

        if not isinstance(password, str):
            error_string = 'password must be a string'
            
            self._logger.error(error_string)
            raise ValueError(error_string)

        pattern = r'^[a-zA-Z0-9_-]+$'
        if not re.match(pattern, password):

            error_string = 'Password contains illegal characters. '\
                            'Only alphanumeric characters, "_" and "-" are allowed.'
            
            self._logger.error(error_string)
            raise ValueError(error_string)


    def _get_rack_grid(self):

        clustering_anomalies = []

        while not self._clustering_queue.empty():
            clustering_anomalies = self._clustering_queue.get()

        html = "<div style='display: grid; grid-template-columns: repeat(10, 40px); grid-gap: 1px;'>"
        for rack_row in _rack_numbers_expected_2023:
            
            for column in range(10):
                if column < len(rack_row):
                    element = rack_row[column]
                    color = "red" if rack_row[column] in clustering_anomalies else "green"
                else:
                    element = ""
                    color = "gray"

                html += f"<div style='width: 40px; height: 40px; background-color: {color};'>" \
                        f"<p style='font-size: 20px; text-align: center; margin-top: 5px;'>{element}</p></div>"
        html += "</div>"
        return html


    def _get_time_series_plot(self):
        with self._time_series_plot_mutex:
             return self._time_series_plot

    
    def _generate_time_series_plot(self):
        while not self._time_series_queue.empty():
            data, timestamps, labels =\
                self._time_series_queue.get()

            data = np.atleast_1d(data)
            timestamps = np.atleast_1d(timestamps)
            labels = np.atleast_1d(labels)

            # self._logger.info(f'{data}')
            # self._logger.info(f'{timestamps}')
            # self._logger.info(f'{labels}')
            
            for data_point, timestamp, label in\
                        zip(data, timestamps, labels):
                self._time_series_buffer.append(
                        (data_point, timestamp, label))
        
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        fig.set_facecolor('#1f2937')
        ax.set_facecolor('#1f2937')

        if len(self._time_series_buffer) > 1:
            data, timestamps, label =\
                tuple(map(list, zip(*list(self._time_series_buffer))))

            data = np.vstack(data)

            data_dims = data.shape[-1]

            data_median = data[:, :data_dims//2]
            data_std = data[:, data_dims//2:]
            
            label = np.array(label, dtype=np.uint8)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%X'))
            
            ax.grid(color='white', axis='y')

            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')

            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            plt.xticks(rotation=30, ha='right')
            # plt.yticks(rotation=30, ha='right')

            ax.set_ylabel('DCM-Rate')
            ax.set_xlabel('Time')

            median_plot = ax.plot(timestamps,
                                    data_median,
                                    linewidth=1.5)

            outline_lower = data_median - data_std
            outline_upper = data_median + data_std

            for channel in range(data_median.shape[-1]):
                ax.fill_between(timestamps,
                                        outline_lower[:, channel],
                                        outline_upper[:, channel],
                                        color=median_plot[channel].get_color(),
                                        linewidth=0,
                                        alpha=0.2)

            anomaly_starts, anomaly_ends =\
                        get_anomalous_runs(label)

            for start, end in zip(anomaly_starts,
                                        anomaly_ends):
                start = max(0, start - 1)
                end = min(end, len(timestamps) - 1)
                
                ax.axvspan(timestamps[start],
                                timestamps[end],
                                color='red', alpha=0.5)

        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Waiting for\nstart of detection...', ha='center', va='center',
                    fontsize=20, color='cyan')

        plt.tight_layout()

        with self._time_series_plot_mutex:
            self._time_series_plot = fig

    
    def _get_log_string(self):

        while not self._log_queue.empty():
            record = self._log_queue.get()

            if (record.levelno >= 10) and\
                not ('http' in record.module) and\
                not ('_client' in record.module):
                message = f'{record.asctime} | {record.module} | {record.levelname}: {record.msg}'

                # message_highlighted =\
                #     _color_text_by_level(message, record.levelno)

                # self._log_buffer.append(message_highlighted)

                self._log_buffer.append(message)

        log_string = ''

        for log in list(self._log_buffer):
            log_string += log + '\n'

        return log_string


    def launch(self):
        self._dashboard.queue().launch(
                        auth=self._auth_data,
                        server_name=self._address)
        
        raise KeyboardInterrupt
