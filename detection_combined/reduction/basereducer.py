#!/usr/bin/env python3

from abc import ABC, abstractmethod
import logging

import numpy as np
from pandas import DataFrame

_rack_numbers_expected =   [16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26,
                            44, 45, 46, 47, 48, 49,
                            50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                            60, 61, 62, 63,
                            70, 71, 72, 73, 74, 75, 76, 77, 79,
                            80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                            90]

class BaseReducer(ABC):

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

        self._missing_racks_feedback_given = False

    @abstractmethod
    def reduce_pandas(self, input_slice: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def reduce_numpy(self,
                        input_slice: np.array,
                        tpu_labels: list,
                        timestamps: list) -> DataFrame:
        pass

    def _adjust_reduced_data(self,
                                labels_reduced: np.array,
                                data_reduced: np.array) -> np.array:

        rack_count_expected = len(_rack_numbers_expected)
        rack_count_observed = len(labels_reduced)//2

        if rack_count_observed < rack_count_expected:
            rack_numbers_observed =\
                    [int(label.removeprefix('m_'))\
                        for label in labels_reduced[:rack_count_observed]]

            missing_racks = np.setdiff1d(_rack_numbers_expected,
                                            rack_numbers_observed)

            indices_missing =\
                    np.nonzero(np.isin(_rack_numbers_expected,
                                                    missing_racks))[0]

            if not self._missing_racks_feedback_given:

                missing_racks_string =\
                    ', '.join(str(missing_racks))\
                        if len(missing_racks) > 1\
                        else str(missing_racks[0])

                self._logger.warning(f'Rack(s) {missing_racks_string} are '
                                        'missing. 2nd stage detection '
                                        'performance might be affected.')

                missing_rack_indices_string =\
                            ', '.join(str(indices_missing))\
                                if len(indices_missing) > 1\
                                else str(indices_missing[0])

                self._logger.debug('Indices missing racks: '
                                    f'{missing_rack_indices_string}')

                self._missing_racks_feedback_given = True

            data_reduced = np.insert(data_reduced,
                                        indices_missing,
                                        0, axis=1)

            data_reduced = np.insert(data_reduced,
                                        indices_missing +\
                                            rack_count_expected,
                                        0, axis=1)

            missing_labels_median =\
                [f'm_{rack}' for rack in missing_racks]

            labels_reduced = np.insert(labels_reduced,
                                        indices_missing,
                                        missing_labels_median)

            missing_labels_std =\
                [f'std_{rack}' for rack in missing_racks]

            labels_reduced = np.insert(labels_reduced,
                                        indices_missing +\
                                            rack_count_expected,
                                        missing_labels_std)

        return labels_reduced, data_reduced
