#!/usr/bin/env python3

from abc import ABC, abstractmethod
import logging

import numpy as np
from pandas import DataFrame

from utils.channellabels import subgroup_labels_expected_hlt_dcm_2018,\
                                    subgroup_labels_expected_hlt_dcm_2023,\
                                    subgroup_labels_expected_eclipse
                                    

class BaseReducer(ABC):

    def __init__(self, configuration_version: str) -> None:
        self._configuration_version = configuration_version

        if self._configuration_version == '2018':
            self._subgroup_numbers_expected = subgroup_labels_expected_hlt_dcm_2018
        elif self._configuration_version == '2023':
            self._subgroup_numbers_expected = subgroup_labels_expected_hlt_dcm_2023
        elif self._configuration_version == 'ECLIPSE':
            self._subgroup_numbers_expected = subgroup_labels_expected_eclipse
        else:
            raise ValueError('Configuration version '
                                f'{self._configuration_version} '
                                'is unknown')

        self._logger = logging.getLogger(__name__)
        self._missing_subgroups_feedback_given = False

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

        subgroup_count_expected = len(self._subgroup_numbers_expected)
        subgroup_count_observed = len(labels_reduced)//2

        if subgroup_count_observed < subgroup_count_expected:
            subgroup_numbers_observed =\
                    [int(label.removeprefix('m_'))\
                        for label in labels_reduced[:subgroup_count_observed]]

            missing_subgroups = np.setdiff1d(self._subgroup_numbers_expected,
                                                subgroup_numbers_observed)

            indices_missing =\
                    np.nonzero(np.isin(self._subgroup_numbers_expected,
                                                    missing_subgroups))[0]

            if not self._missing_subgroups_feedback_given:
                
                missing_subgroups_string = ''

                for subgroup in missing_subgroups:
                    missing_subgroups_string += f'{subgroup}, '

                self._logger.warning(f'Rack(s) {missing_subgroups_string} are '
                                        'missing. 2nd stage detection '
                                        'performance might be affected.')

                # missing_subgroup_indices_string =\
                #             ', '.join(str(indices_missing))\
                #                 if len(indices_missing) > 1\
                #                 else str(indices_missing[0])

                # self._logger.debug('Indices missing subgroups: '
                #                     f'{missing_subgroup_indices_string}')

                self._missing_subgroups_feedback_given = True

            data_reduced = np.insert(data_reduced,
                                        indices_missing,
                                        0, axis=1)

            data_reduced = np.insert(data_reduced,
                                        indices_missing +\
                                            subgroup_count_expected,
                                        0, axis=1)

            missing_labels_median =\
                [f'm_{subgroup}' for subgroup in missing_subgroups]

            labels_reduced = np.insert(labels_reduced,
                                        indices_missing,
                                        missing_labels_median)

            missing_labels_std =\
                [f'std_{subgroup}' for subgroup in missing_subgroups]

            labels_reduced = np.insert(labels_reduced,
                                        indices_missing +\
                                            subgroup_count_expected,
                                        missing_labels_std)

        return labels_reduced, data_reduced
