#!/usr/bin/env python3

from enum import Enum

class AnomalyType(Enum):
        ClusteringDropToZero = 0
        ClusteringDropout = 1
        ClusteringGeneral = 2
        TransformerBased = 3


class RunAnomaly():
    def __init__(self):
        self.duration = 0
        self.anomaly_types = []

    def to_json(self) -> str:

        json_dict = {'duration': self.duration}
        json_dict['types'] = [anomaly_type.name for anomaly_type in self.anomaly_types]

        return json_dict

    def update(self, duration, type):
        self.duration = duration

        if len(self.anomaly_types) != 0:
            if self.anomaly_types[-1] != type:
                self.anomaly_types.append(type)
        else:
            self.anomaly_types.append(type)