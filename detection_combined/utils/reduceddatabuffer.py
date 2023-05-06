
from collections.abc import Callable
from collections import deque

import pandas as pd

class ReducedDataBuffer():
    def __init__(self,
                    size: int,
                    buffer_filled_callback: Callable = None) -> None:

        self._size = size
        self._buffer_filled_callback =\
                    buffer_filled_callback

        self._buffer = deque([], maxlen=self._size)


    def push(self, data: pd.DataFrame):
        self._buffer.append(data)

        # print(self._buffer)

        if len(self._buffer) == self._size:

            buffer_list = list(self._buffer)

            buffer_pd = pd.concat((buffer_list))

            if not isinstance(self._buffer_filled_callback, type(None)):
                return self._buffer_filled_callback(buffer_pd)


    def set_buffer_filled_callback(self,
                                    callback: Callable) -> None:
        self._buffer_filled_callback = callback