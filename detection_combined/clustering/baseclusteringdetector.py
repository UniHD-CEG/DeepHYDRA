
from abc import ABC
from collections.abc import Callable
import logging

class BaseClusteringDetector(ABC):

    def __init__(self) -> None:
        self.detection_callback = None

        self._logger = logging.getLogger(__name__)

    def register_detection_callback(self,
                                        callback: Callable) -> None:
        self.detection_callback = callback