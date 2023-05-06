from concurrent.futures import process
import os
import math
from multiprocessing import Pool
from functools import partial

import numpy as np
import numpy.lib.stride_tricks as np_st
import pandas as pd
from scipy.fft import fft, ifft
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

_implemented_augmentations = ['Scale',
                                'APP',
                                'Scale_APP']


def _get_runs_of_true(x):
    """Find runs of consecutive items in an array.
        As published in https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065"""

    # Ensure array

    x = np.asanyarray(x)

    if x.ndim != 1:
        raise ValueError('Only 1D arrays supported')

    n = x.shape[0]

    # Handle empty array

    if n == 0:
        return np.array([]), np.array([])

    else:

        # Find run starts

        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True

        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # Find run values
        run_values = x[loc_run_start]

        # Find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        run_starts = np.compress(run_values, run_starts)
        run_lengths = np.compress(run_values, run_lengths)

        return run_starts, run_lengths
    

class HLTDataTimeseriesAugmentor():

    def __init__(self, applied_augmentations: list) -> None:

        self.applied_augmentations = applied_augmentations

        for augmentation, parameters in self.applied_augmentations:
            if augmentation not in _implemented_augmentations:
                raise NotImplementedError(
                        f'Augmentation {augmentation} not implemented')

            if augmentation == 'Scale':
                if not isinstance(parameters, list):
                    raise ValueError('Invalid argument for '
                                        'scaling augmentation')

                if len(parameters) != 2:
                    raise ValueError('Invalid number of scaling '
                                        'augmentation parameters')

                if not (np.issubdtype(type(parameters[0]), np.number) and\
                                np.issubdtype(type(parameters[1]), np.number)):
                     raise ValueError('Scaling range values must be numbers')


            elif augmentation == 'APP':
                if not isinstance(parameters, list):
                    raise ValueError('Invalid argument for '
                                            'APP augmentation')

                if len(parameters) != 3:
                    raise ValueError('Invalid number of APP '
                                        'augmentation parameters')

                if np.issubdtype(type(parameters[0]), np.floating):
                    if parameters[0] <= 0 or\
                            parameters[0] > 1:
                        raise ValueError('APP segment length ratio '
                                            'must be in range (0, 1]')

                else:
                     raise ValueError('APP segment length ration '
                                        'must be a floating point number')

                if (np.issubdtype(type(parameters[1]), np.number) and\
                                np.issubdtype(type(parameters[2]), np.number)):
                    if parameters[1] < 0:
                        raise ValueError('APP amplitude perturbation '
                                            'factor must be greater than 0')
                    if parameters[2] < 0:
                        raise ValueError('APP phase perturbation '
                                            'factor must be greater than 0')

                else:
                     raise ValueError('Amplitude and phase perturbation '
                                                'amounts must be numbers')

        self.rng = np.random.default_rng(seed=42)


    def _find_collision_windows(self,
                                    data: np.array,
                                    downtime_period: int,
                                    tolerance_abs: float) -> list:

        row_close_to_zero = [np.allclose(row, 0, tolerance_abs)\
                                            for row in data.tolist()]

        window_starts, window_lengths =\
                _get_runs_of_true(row_close_to_zero)

        collision_window_ends = [window_start + window_length//2\
                                    for window_start, window_length in\
                                        zip(window_starts, window_lengths)\
                                            if window_length >= downtime_period]

        collision_window_ends.append(len(data))

        return collision_window_ends

    def _segment_dataframe(self,
                            data: pd.DataFrame,
                            downtime_period: int,
                            tolerance_abs: float) -> list:

        collision_window_ends =\
                self._find_collision_windows(data.to_numpy(),
                                                downtime_period,
                                                tolerance_abs)


        segment_list = []

        collision_window_start = 0

        for collision_window_end in collision_window_ends:
            segment_list.append(
                data.iloc[collision_window_start:collision_window_end, :])

            collision_window_start = collision_window_end

        pd.testing.assert_frame_equal(data,
                                        pd.concat(segment_list))

        return segment_list

    def _segment_dataframe_labeled(self,
                                    data: pd.DataFrame,
                                    label: pd.DataFrame,
                                    downtime_period: int,
                                    tolerance_abs: float) -> list:

        collision_window_ends =\
                self._find_collision_windows(data.to_numpy(),
                                                downtime_period,
                                                tolerance_abs)

        segment_list_data = []
        segment_list_label = []

        collision_window_start = 0

        for collision_window_end in collision_window_ends:
            segment_list_data.append(
                data.iloc[collision_window_start:collision_window_end, :])
            segment_list_label.append(
                label.iloc[collision_window_start:collision_window_end, :])

            collision_window_start = collision_window_end

        pd.testing.assert_frame_equal(data,
                                        pd.concat(segment_list_data))

        return segment_list_data, segment_list_label


    def _adjust_index(self,
                        index_previous: pd.DatetimeIndex,
                        index_current: pd.DatetimeIndex) -> pd.DatetimeIndex:
        
        index_current = pd.Series(index_current)

        index_current -= index_current[0] - index_previous[-1]
        index_current += pd.Timedelta(5, unit='s')

        return pd.DatetimeIndex(index_current)


    def _prepend_zeros(self,
                        data: pd.DataFrame,
                        count_min: int,
                        count_max: int) -> pd.DataFrame:

        count = self.rng.integers(count_min,
                                    count_max,
                                    endpoint=True)

        zeros = np.zeros((count, data.shape[1]))

        stdev_index_original =\
                pd.Series(data.index[:-1] - data.index[1:]).std()

        start_datetime_zeros = data.index[0] -\
                                pd.Timedelta(len(zeros)*5, unit='S')

        index_zeros = pd.Series(pd.date_range(
                                    start=start_datetime_zeros,
                                    end=data.index[0],
                                    freq='5S',
                                    inclusive='left'))

        # Add jitter to synthetic timestamps
        # to mimic timestamp distribution of
        # original data

        for count in range(len(index_zeros)):
            jitter = self.rng.normal(scale=stdev_index_original.delta)
            index_zeros[count] += pd.Timedelta(jitter,
                                                unit='ns')

        index_zeros = pd.DatetimeIndex(index_zeros)

        data = pd.concat((pd.DataFrame(zeros,
                                        index=index_zeros,
                                        columns=data.columns),
                                        data))

        return data


    def _prepend_zeros_labeled(self,
                                data: pd.DataFrame,
                                label: pd.DataFrame,
                                count_min: int,
                                count_max: int) -> pd.DataFrame:

        count = self.rng.integers(count_min,
                                    count_max,
                                    endpoint=True)

        zeros = np.zeros((count, data.shape[1]))

        stdev_index_original =\
                pd.Series(data.index[:-1] - data.index[1:]).std()

        start_datetime_zeros = data.index[0] -\
                                pd.Timedelta(len(zeros)*5, unit='S')

        index_zeros = pd.Series(pd.date_range(
                                    start=start_datetime_zeros,
                                    end=data.index[0],
                                    freq='5S',
                                    inclusive='left'))

        # Add jitter to synthetic timestamps
        # to mimic timestamp distribution of
        # original data

        for count in range(len(index_zeros)):
            jitter = self.rng.normal(scale=stdev_index_original.delta)
            index_zeros[count] += pd.Timedelta(jitter,
                                                unit='ns')

        index_zeros = pd.DatetimeIndex(index_zeros)

        data = pd.concat((pd.DataFrame(zeros,
                                        index=index_zeros,
                                        columns=data.columns),
                                        data))

        label = pd.concat((pd.DataFrame(zeros,
                                        index=index_zeros,
                                        columns=label.columns),
                                        label))

        return data, label


    def _amplitude_phase_perturbation(self,
                                        data: np.array,
                                        segment_length_ratio: float = 0.1,
                                        amp_perturbation_amount: float = 0.5,
                                        phase_perturbation_amount: float = 0.1) -> np.array:

        """
        As described in "RobustTAD: Robust Time Series Anomaly 
        Detection via Decomposition and Convolutional Neural Networks"
        by Gao et al.
        """

        for column in range(data.shape[1]):

            channel_data_f = fft(data[:, column])

            amp_data = np.abs(channel_data_f)
            phase_data = np.angle(channel_data_f)

            segment_length =\
                    max(math.floor(segment_length_ratio*\
                                            len(amp_data)), 2)

            stride = math.ceil(segment_length/2)

            sliding_window_amp_data =\
                np_st.sliding_window_view(amp_data,
                                            segment_length,
                                            writeable=True)[::stride, :]

            sliding_window_phase_data =\
                np_st.sliding_window_view(phase_data,
                                            segment_length,
                                            writeable=True)[::stride, :]

            assert sliding_window_amp_data.shape ==\
                            sliding_window_phase_data.shape

            for window in range(len(sliding_window_amp_data)):

                # Replace magnitude values in window with
                # Gaussian noise with the same mean and
                # variance as the original magnitude values

                amp_data_window_mean =\
                    np.mean(sliding_window_amp_data[window, :])

                amp_data_window_std =\
                    np.std(sliding_window_amp_data[window, :])

                gaussian_noise_amp =\
                        self.rng.normal(0, amp_data_window_std,
                                            size=(segment_length,))

                sliding_window_amp_data[window, :] +=\
                                            amp_perturbation_amount*\
                                            gaussian_noise_amp

                # Perturb phase by adding Gaussian noise of
                # the same standard deviation as the window

                phase_data_window_std =\
                    np.std(sliding_window_phase_data[window, :])

                gaussian_noise_phase =\
                        self.rng.normal(0, phase_data_window_std,
                                            size=(segment_length,))

                sliding_window_phase_data[window, :] +=\
                                        phase_perturbation_amount*\
                                        gaussian_noise_phase

            e_i_phase = np.exp(phase_data*1j)

            data[:, column] =\
                np.maximum(0, ifft(amp_data*e_i_phase))

        return data


    def fit_transform(self,
                        data: pd.DataFrame,
                        target_size_min: int,
                        ratio_augmented: float) -> pd.DataFrame:

        segments_unmodified = self._segment_dataframe(data, 5, 0.5)

        segments_augmented = []
        augmented_dataset_size = 0

        # Probability that a selected augmentation or no
        # augmentation will be applied. All augmentations
        # are equally likely to to be applied, with a 
        # probability of ratio_augmented/len(self.applied_augmentations).
        # The probability that no augmentation is applied
        # is 1 - ratio_augmented.

        p_augmentation = np.full(len(self.applied_augmentations) + 1,
                                    ratio_augmented/len(self.applied_augmentations))

        p_augmentation[-1] = 1 - ratio_augmented

        while augmented_dataset_size < target_size_min:

            index_source_segment =\
                    self.rng.choice(len(segments_unmodified))

            source_segment =\
                segments_unmodified[index_source_segment].copy()

            index_applied_augmentation =\
                    self.rng.choice(len(self.applied_augmentations) + 1,
                    p=p_augmentation)

            if index_applied_augmentation >=\
                    len(self.applied_augmentations):

                source_segment = self._prepend_zeros(source_segment,
                                                            64, 256)

                # Adjust indices to guarantee the combined 
                # index is strictly increasing

                if len(segments_augmented) != 0:
                    source_segment.index =\
                            self._adjust_index(segments_augmented[-1].index,
                                                            source_segment.index)

                segments_augmented.append(source_segment)

            else:

                source_segment_np = source_segment.to_numpy()

                augmentation, parameters =\
                        self.applied_augmentations[
                                index_applied_augmentation]

                segment_augmented_np = None
                segment_augmented_pd = None

                if augmentation == 'Scale':
                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    segment_augmented_np = source_segment_np*\
                                                scaling_factor

                elif augmentation == 'APP':

                    segment_length_ratio = parameters[0]
                    amp_perturbation_amount = parameters[1]
                    phase_perturbation_amount = parameters[2]

                    segment_augmented_np =\
                        self._amplitude_phase_perturbation(source_segment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                elif augmentation == 'Scale_APP':

                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    source_segment_np = source_segment_np*\
                                                scaling_factor

                    segment_length_ratio = parameters[2]
                    amp_perturbation_amount = parameters[3]
                    phase_perturbation_amount = parameters[4]

                    segment_augmented_np =\
                        self._amplitude_phase_perturbation(source_segment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                segment_augmented_pd = pd.DataFrame(segment_augmented_np,
                                                        source_segment.index,
                                                        columns=source_segment.columns)

                segment_augmented_pd =\
                        self._prepend_zeros(segment_augmented_pd,
                                                            64, 256)

                # Adjust indices to guarantee the combined 
                # index is strictly increasing

                if len(segments_augmented) != 0:
                    segment_augmented_pd.index =\
                                self._adjust_index(segments_augmented[-1].index,
                                                        segment_augmented_pd.index)    
                    
                segments_augmented.append(segment_augmented_pd)

            augmented_dataset_size += len(source_segment)

        dataset_augmented = pd.concat(segments_augmented)

        return dataset_augmented


    def fit_transform_labeled(self,
                                data: pd.DataFrame,
                                label: pd.DataFrame,
                                target_size_min: int,
                                ratio_augmented: float) -> pd.DataFrame:


        data_segments_unmodified,\
            label_segments_unmodified =\
                self._segment_dataframe_labeled(data, label, 5, 0.5)

        data_segments_augmented = []
        label_segments_augmented = []

        augmented_dataset_size = 0

        # Probability that a selected augmentation or no
        # augmentation will be applied. All augmentations
        # are equally likely to to be applied, with a 
        # probability of ratio_augmented/len(self.applied_augmentations).
        # The probability that no augmentation is applied
        # is 1 - ratio_augmented.

        p_augmentation = np.full(len(self.applied_augmentations) + 1,
                                    ratio_augmented/len(self.applied_augmentations))

        p_augmentation[-1] = 1 - ratio_augmented

        while augmented_dataset_size < target_size_min:

            index_source_segment =\
                    self.rng.choice(len(data_segments_unmodified))

            data_source_segment =\
                data_segments_unmodified[index_source_segment].copy()

            label_source_segment =\
                label_segments_unmodified[index_source_segment].copy()

            index_applied_augmentation =\
                    self.rng.choice(len(self.applied_augmentations) + 1,
                    p=p_augmentation)

            if index_applied_augmentation >=\
                    len(self.applied_augmentations):

                data_source_segment,\
                    label_source_segment =\
                        self._prepend_zeros_labeled(data_source_segment,
                                                        label_source_segment,
                                                        64, 256)

                # Adjust indices to guarantee the combined 
                # index is strictly increasing

                if len(data_segments_augmented) != 0:
                    data_source_segment.index =\
                            self._adjust_index(data_segments_augmented[-1].index,
                                                            data_source_segment.index)

                    label_source_segment.index =\
                            self._adjust_index(label_segments_augmented[-1].index,
                                                            label_source_segment.index)

                data_segments_augmented.append(data_source_segment)
                label_segments_augmented.append(label_source_segment)

            else:

                data_source_segment_np =\
                    data_source_segment.to_numpy()

                augmentation, parameters =\
                        self.applied_augmentations[
                                index_applied_augmentation]

                data_segment_augmented_np = None
                data_segment_augmented_pd = None

                if augmentation == 'Scale':
                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    data_segment_augmented_np = data_source_segment_np*\
                                                scaling_factor

                elif augmentation == 'APP':

                    segment_length_ratio = parameters[0]
                    amp_perturbation_amount = parameters[1]
                    phase_perturbation_amount = parameters[2]

                    data_segment_augmented_np =\
                        self._amplitude_phase_perturbation(data_source_segment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                elif augmentation == 'Scale_APP':

                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    data_source_segment_np = data_source_segment_np*\
                                                scaling_factor

                    segment_length_ratio = parameters[2]
                    amp_perturbation_amount = parameters[3]
                    phase_perturbation_amount = parameters[4]

                    data_segment_augmented_np =\
                        self._amplitude_phase_perturbation(data_source_segment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                data_segment_augmented_pd = pd.DataFrame(data_segment_augmented_np,
                                                            data_source_segment.index,
                                                            columns=data_source_segment.columns)

                data_segment_augmented_pd,\
                    label_segment_augmented_pd =\
                        self._prepend_zeros_labeled(data_segment_augmented_pd,
                                                            label_source_segment,
                                                                            64, 256)

                # Adjust indices to guarantee the combined 
                # index is strictly increasing

                if len(data_segments_augmented) != 0:

                    data_segment_augmented_pd.index =\
                                self._adjust_index(data_segments_augmented[-1].index,
                                                        data_segment_augmented_pd.index)

                    label_segment_augmented_pd.index =\
                                        self._adjust_index(label_segments_augmented[-1].index,
                                                                label_segment_augmented_pd.index)

                    
                data_segments_augmented.append(data_segment_augmented_pd)
                label_segments_augmented.append(label_segment_augmented_pd)

            augmented_dataset_size += len(data_source_segment)

        dataset_augmented = pd.concat(data_segments_augmented)
        labels_augmented = pd.concat(label_segments_augmented)

        return dataset_augmented, labels_augmented