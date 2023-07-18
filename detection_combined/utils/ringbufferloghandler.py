
import logging
from collections.abc import Callable
from collections import deque


class RingbufferLogHandler(logging.Handler):
    def __init__(self,
                    callback: Callable,
                    size: int) -> None:

        self._size = size
        self._callback = callback

        self._buffer = deque([], maxlen=self._size)


    def emit(self, record):
        self._buffer.append(record)

        buffer_list = list(self._buffer)

        if not isinstance(self._callback, type(None)):
            self._callback(buffer_list)
            

    def set_callback(self, callback: Callable) -> None:
        self._callback = callback