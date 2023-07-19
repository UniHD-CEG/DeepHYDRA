import math
import re
import multiprocessing as mp
import ipaddress
import logging
from collections import deque
from collections.abc import Iterable

import pandas as pd

import gradio as gr
import datetime
import numpy as np


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


def _color_text_by_level(text: str,
                            level: int):
    
    if level <= 20:
        color_string = '\033[32m'
    elif level == 30:
        color_string = '\033[33m'
    else:
        color_string = '\033[31m'

    return f'\033[1m{color_string}{text}\033[0m'



class GradioServer():
    def __init__(self,
                    data_queue: mp.Queue,
                    log_queue: mp.Queue,
                    address: str = 'localhost',
                    auth_data = None) -> None:

        self._data_queue = data_queue
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

        self._log_buffer = deque([], maxlen=25)

        self._dashboard = gr.Blocks(analytics_enabled=False)

        with self._dashboard:
            with gr.Row():
                with gr.Column():
                    self._plot = gr.LinePlot(show_label=False)
                with gr.Column():
                    with gr.Row():
                        gr.Textbox('Cluster Status',
                                            label='')
                    with gr.Row():
                        cluster_status = gr.HTML()
            with gr.Row():
                self._log_output = gr.Textbox(label='Log Output',
                                                        lines=25,
                                                        interactive=False)

            self._dashboard.load(lambda: self._create_rack_grid(_rack_status),
                                                            None, cluster_status)
            
            self._dashboard.load(self.get_log_string, None,
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


    def _create_rack_grid(self, rack_states):
        html = "<div style='display: grid; grid-template-columns: repeat(10, 50px); grid-gap: 1px;'>"
        for rack_row, rack_row_states in zip(
                                    _rack_numbers_expected_2023,
                                    rack_states):
            
            for column in range(10):
                if column < len(rack_row):
                    element = rack_row[column]
                    color = "red" if rack_row_states[column] else "green"
                else:
                    element = ""
                    color = "gray"

                html += f"<div style='width: 50px; height: 50px; background-color: {color};'>" \
                        f"<p style='font-size: 25px; text-align: center; margin-top: 5px;'>{element}</p></div>"
        html += "</div>"
        return html


    def launch(self):
        self._dashboard.queue().launch(
                        auth=self._auth_data,
                        server_name=self._address)
        
    
    def update_plot(self):
        while not self._data_queue.empty():
            data, label = self._data_queue.get()
        
        update = gr.LinePlot.update(
        value=pd.DataFrame({"x": x, "y": y}),
        x='Time',
        y='DCM-Rate',
        title='DCM-Rate',
        width=600,
        height=350)

        return update

    
    def get_log_string(self):

        while not self._log_queue.empty():
            record = self._log_queue.get()

            if record.level >= 20:
                message = f'{record.levelname}: {record.msg}'

                message_highlighted =\
                    _color_text_by_level(message, record.level)

                self._log_buffer.append(message_highlighted)

        log_string = ''

        for log in list(self._log_buffer):
            log_string += log + '\n'

        return log_string






