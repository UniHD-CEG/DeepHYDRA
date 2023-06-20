#!/usr/bin/env python3

import time as t
import datetime as dt

from .singleton import Singleton


class ShiftedTimeSingleton(metaclass=Singleton):   
    def __init__(self,
                    start_datetime: dt.datetime) -> None:
        self._timedelta = dt.datetime.now() - start_datetime

    def now(self) -> None:
        return dt.datetime.now() - self._timedelta
