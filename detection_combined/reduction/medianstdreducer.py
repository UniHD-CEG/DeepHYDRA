#!/usr/bin/env python3

import re
from collections import defaultdict
import logging

import numpy as np
import pandas as pd

from .basereducer import BaseReducer
from utils.tqdmloggingdecorator import tqdmloggingdecorator


class MedianStdReducer(BaseReducer):

    def __init__(self) -> None:
        super(MedianStdReducer, self).__init__()

        self._columns_reduced = None
        self._keys_last = None


    def _parse_channel_name(self, channel_name):
        parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
        return parameters[1]


    def _create_channel_names(self,
                                median_labels,
                                stdev_labels):

        median_labels = ['m_{}'.format(median_label)\
                            for median_label in median_labels]

        stdev_labels = ['std_{}'.format(stdev_label)
                            for stdev_label in stdev_labels]

        labels = np.concatenate((median_labels,
                                    stdev_labels))

        return labels

    @tqdmloggingdecorator
    def reduce_numpy(self,
                        machine_labels: list,
                        timestamps: list,
                        input_slice: np.array) -> pd.DataFrame:

        subgroup_numbers = [self._parse_channel_name(label) for label in machine_labels]

        # Reduce input slice

        slice_reduced_list = []

        for row_x_data in np.atleast_2d(input_slice):

            subgroup_buckets_data = defaultdict(list)

            for index, datapoint in enumerate(row_x_data):
                subgroup_buckets_data[subgroup_numbers[index]].append(datapoint)

            subgroup_median_hlt = {}
            subgroup_hlt_stdevs = {}

            for subgroup, subgroup_bucket in subgroup_buckets_data.items():
                
                subgroup_median_hlt[subgroup] = np.nanmedian(subgroup_bucket)
                subgroup_hlt_stdevs[subgroup] = np.nanstd(subgroup_bucket)

            subgroup_median_hlt = dict(sorted(subgroup_median_hlt.items()))
            subgroup_hlt_stdevs = dict(sorted(subgroup_hlt_stdevs.items()))

            if not isinstance(self._keys_last, type(None)):
                if not (subgroup_median_hlt.keys() == self._keys_last):
                    error_message_line_0 =\
                        'Subgroup bucket keys changed between slices'
                    error_message_line_1 =\
                        f'Previous keys: {self._keys_last}\t'
                    error_message_line_2 =\
                        f'Current keys: {subgroup_median_hlt.keys()}'

                    non_intersecting_keys =\
                            list(set(self._keys_last) ^\
                            set(subgroup_median_hlt.keys()))

                    error_message_line_3 =\
                        f'Keys not in both slices: {non_intersecting_keys}'

                    self._logger.error(error_message_line_0)
                    self._logger.debug(error_message_line_1)
                    self._logger.debug(error_message_line_2)
                    self._logger.debug(error_message_line_3)

                    raise RuntimeError(error_message_line_0)

                if not (subgroup_median_hlt.keys() == subgroup_hlt_stdevs.keys()):
                    error_message_line_0 =\
                        'Subgroup bucket keys not identical between '\
                        'Median and Stdev Buckets'
                    error_message_line_1 =\
                        f'Median keys: {subgroup_median_hlt.keys()}\t'
                    error_message_line_2 =\
                        f'Stdev keys: {subgroup_hlt_stdevs.keys()}'

                    non_intersecting_keys =\
                            list(set(subgroup_median_hlt.keys()) ^\
                                        set(subgroup_hlt_stdevs.keys()))

                    error_message_line_3 =\
                        f'Keys not in both: {non_intersecting_keys}'

                    self._logger.error(error_message_line_0)

                    self._logger.debug(error_message_line_1)
                    self._logger.debug(error_message_line_2)
                    self._logger.debug(error_message_line_3)

                    raise RuntimeError(error_message_line_0)

            self._keys_last = subgroup_median_hlt.keys()

            if isinstance(self._columns_reduced, type(None)):
                self._columns_reduced =\
                            self._create_channel_names(subgroup_median_hlt.keys(),
                                                            subgroup_hlt_stdevs.keys())

            subgroup_data_np = np.concatenate((np.array(list(subgroup_median_hlt.values())),
                                                np.array(list(subgroup_hlt_stdevs.values()))))

            slice_reduced_list.append(subgroup_data_np)

        slice_reduced_np = np.stack(slice_reduced_list)
        slice_reduced_np = np.nan_to_num(slice_reduced_np, nan=-1)

        # nan_amount_reduced = 100*pd.isna(slice_reduced_np.flatten()).sum()/\
        #                                                         slice_reduced_np.size

        # self._logger.debug('NaN amount reduced slice: {:.2f} %'.format(nan_amount_reduced))

        timestamps = np.atleast_1d(np.asanyarray(timestamps))

        columns_reduced_adjusted,\
                slice_reduced_np =\
                    self._adjust_reduced_data(
                                    self._columns_reduced,
                                    slice_reduced_np)

        result_slice = pd.DataFrame(slice_reduced_np,
                                            timestamps,
                                            columns_reduced_adjusted)

        return result_slice


    def reduce_pandas(self,
                        input_slice: pd.DataFrame) -> pd.DataFrame:

        machine_labels = list((input_slice).columns.values)
        timestamps = list(input_slice.index)

        input_slice_np = input_slice.to_numpy()

        return self.reduce_numpy(machine_labels,
                                    timestamps,
                                    input_slice_np)