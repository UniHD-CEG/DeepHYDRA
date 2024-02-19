
import logging
from collections.abc import Callable
from collections import deque

import numpy as np
import pandas as pd


class ReducedDataBuffer():
    def __init__(self,
                    size: int,
                    buffer_filled_callback: Callable = None) -> None:

        self._size = size
        self._buffer_filled_callback =\
                    buffer_filled_callback

        self._buffer = deque([], maxlen=self._size)

        self._logger = logging.getLogger(__name__)
        self._buffer_filled_feedback_given = False


    def push(self, data: pd.DataFrame):
        
        data_size = len(data)

        if data_size > 1:
            for data_row in np.vsplit(data, data_size):
                self._buffer.append(data_row)

        elif data_size == 1:
            self._buffer.append(data)
        else:
            return

        # print(self._buffer)

        if len(self._buffer) == self._size:

            buffer_list = list(self._buffer)

            buffer_pd = pd.concat((buffer_list))

            if not isinstance(self._buffer_filled_callback, type(None)):
                return self._buffer_filled_callback(buffer_pd)

            if not self._buffer_filled_feedback_given:
                self._logger.info('Buffer filled')
                self._buffer_filled_feedback_given = True


    def set_buffer_filled_callback(self,
                                    callback: Callable) -> None:
        self._buffer_filled_callback = callback