from concurrent.futures import process
import os
import math
from multiprocessing import Pool
from functools import partial, reduce

import numpy as np
import numpy.lib.stride_tricks as np_st
import pandas as pd
from scipy.fft import fft, ifft
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update({'font.size': 12})

_implemented_augmentations_hlt = ['Scale',
                                'Dilate',
                                'APP',
                                'Scale_APP',
                                'IAAFT',
                                'Scale_IAAFT']

_implemented_augmentations_eclipse = ['Identity',
                                        'Scale',
                                        'APP',
                                        'Scale_APP',
                                        'IAAFT',
                                        'Scale_IAAFT',
                                        'Roll',
                                        'Roll_Scale',
                                        'Roll_APP',
                                        'Roll_Scale_APP',
                                        'Roll_IAAFT',
                                        'Roll_Scale_IAAFT']


# These are the ordinal columns of the preprocessed
# Eclipse dataset, and will thus be excluded from augmentation.

_ordinal_cols_eclipse = ('exa_meminfo_commitlimit_6',
                            'exa_meminfo_dirty_6',
                            'exa_meminfo_hardwarecorrupted_6',
                            'exa_meminfo_writeback_6',
                            'exa_vmstat_drop_pagecache_6',
                            'exa_vmstat_drop_slab_6',
                            'exa_vmstat_nr_anon_transparent_hugepages_6',
                            'exa_vmstat_nr_dirty_6',
                            'exa_vmstat_nr_isolated_file_6',
                            'exa_vmstat_nr_vmscan_immediate_reclaim_6',
                            'exa_vmstat_nr_vmscan_write_6',
                            'exa_vmstat_nr_writeback_6',
                            'exa_vmstat_numa_foreign_6',
                            'exa_vmstat_numa_miss_6',
                            'exa_vmstat_slabs_scanned_6',
                            'lammps_meminfo_commitlimit_6',
                            'lammps_meminfo_dirty_6',
                            'lammps_meminfo_hardwarecorrupted_6',
                            'lammps_meminfo_writeback_6',
                            'lammps_vmstat_drop_pagecache_6',
                            'lammps_vmstat_drop_slab_6',
                            'lammps_vmstat_nr_anon_transparent_hugepages_6',
                            'lammps_vmstat_nr_dirty_6',
                            'lammps_vmstat_nr_isolated_anon_6',
                            'lammps_vmstat_nr_isolated_file_6',
                            'lammps_vmstat_nr_vmscan_immediate_reclaim_6',
                            'lammps_vmstat_nr_vmscan_write_6',
                            'lammps_vmstat_nr_writeback_6',
                            'sw4_meminfo_commitlimit_6',
                            'sw4_meminfo_dirty_6',
                            'sw4_meminfo_hardwarecorrupted_6',
                            'sw4_meminfo_writeback_6',
                            'sw4_vmstat_drop_pagecache_6',
                            'sw4_vmstat_drop_slab_6',
                            'sw4_vmstat_nr_anon_transparent_hugepages_6',
                            'sw4_vmstat_nr_dirty_6',
                            'sw4_vmstat_nr_isolated_anon_6',
                            'sw4_vmstat_nr_isolated_file_6',
                            'sw4_vmstat_nr_vmscan_immediate_reclaim_6',
                            'sw4_vmstat_nr_vmscan_write_6',
                            'sw4_vmstat_nr_writeback_6',
                            'sw4_vmstat_numa_foreign_6',
                            'sw4_vmstat_numa_miss_6',
                            'sw4_vmstat_slabs_scanned_6',
                            'sw4lite_meminfo_commitlimit_6',
                            'sw4lite_meminfo_dirty_6',
                            'sw4lite_meminfo_hardwarecorrupted_6',
                            'sw4lite_meminfo_writeback_6',
                            'sw4lite_vmstat_drop_pagecache_6',
                            'sw4lite_vmstat_drop_slab_6',
                            'sw4lite_vmstat_nr_anon_transparent_hugepages_6',
                            'sw4lite_vmstat_nr_dirty_6',
                            'sw4lite_vmstat_nr_isolated_anon_6',
                            'sw4lite_vmstat_nr_isolated_file_6',
                            'sw4lite_vmstat_nr_vmscan_write_6',
                            'sw4lite_vmstat_nr_writeback_6',
                            'sw4lite_vmstat_numa_foreign_6',
                            'sw4lite_vmstat_numa_interleave_6',
                            'sw4lite_vmstat_numa_miss_6',
                            'sw4lite_vmstat_slabs_scanned_6')




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
            if augmentation not in _implemented_augmentations_hlt:
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

        for count, collision_window_end in enumerate(collision_window_ends):
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


    def _iaaft(self,
                data: np.array,
                sliding_window_size: int = 32,
                tolerance_percent: float = 5,
                candidate_count: int = 10,
                gradient_threshold: float = 2,
                gradient_mask_patience: int = 10) -> np.array:

        usable_processors = len(os.sched_getaffinity(0))
        process_pool = Pool(usable_processors//2)

        data_list_in = np.hsplit(data, data.shape[1])

        _apply_iaaft_with_args =\
                partial(_apply_iaaft,
                            sliding_window_size=sliding_window_size,
                            tolerance_percent=tolerance_percent,
                            candidate_count=candidate_count,
                            gradient_threshold=gradient_threshold,
                            gradient_mask_patience=gradient_mask_patience)
                            

        data_list_out = process_pool.map(_apply_iaaft_with_args,
                                                    data_list_in)

        process_pool.close()

        data = np.hstack(data_list_out)
            
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

            # Sample segment with good data range
            
            # index_source_segment = 16

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


class EclipseDataTimeseriesAugmentor():

    def __init__(self, applied_augmentations: list) -> None:

        self.applications = ['exa', 'lammps', 'sw4', 'sw4lite']

        self.applied_augmentations = applied_augmentations

        for augmentation, parameters in self.applied_augmentations:
            if augmentation not in _implemented_augmentations_eclipse:
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


    def _adjust_index(self,
                        index_previous: pd.DatetimeIndex,
                        index_current: pd.DatetimeIndex) -> pd.DatetimeIndex:
        
        index_current = pd.Series(index_current)

        index_current -= index_current[0] - index_previous[-1]
        index_current += pd.Timedelta(1, unit='s')

        return pd.DatetimeIndex(index_current)


    def _roll_data(self,
                    data: pd.DataFrame,
                    amount_mean: float = 0.,
                    amount_std: float = 1.) -> list:

        for app in self.applications:

            # print(app)

            cols = np.logical_or(data.columns.str.startswith(f'm_{app}_'),
                                    data.columns.str.startswith(f'std_{app}_'))
            
            app_data_np = data.loc[:, cols].to_numpy()

            # print(data)

            roll_amount = self.rng.normal(amount_mean,
                                            amount_std)

            roll_amount = int(roll_amount*len(app_data_np))

            # print(roll_amount)

            app_data_np = np.roll(app_data_np,
                                        roll_amount,
                                        axis=0)

            data.loc[:, cols] = app_data_np

            # print(data)

        return data


    def _roll_data_labeled(self,
                            data: pd.DataFrame,
                            label: pd.DataFrame,
                            amount_mean: float = 0.,
                            amount_std: float = 1.) -> list:

        for app in self.applications:

            # print(app)

            cols_data = np.logical_or(data.columns.str.startswith(f'm_{app}_'),
                                        data.columns.str.startswith(f'std_{app}_'))

            cols_label = label.columns.str.contains(app)

            # print(cols_label)
            
            app_data_np = data.loc[:, cols_data].to_numpy()
            label_np = label.loc[:, cols_label].to_numpy()

            # print(label)

            roll_amount = self.rng.normal(amount_mean,
                                            amount_std)

            roll_amount = int(roll_amount*len(app_data_np))

            # print(roll_amount)

            app_data_np = np.roll(app_data_np,
                                        roll_amount,
                                        axis=0)
            
            label_np = np.roll(label_np,
                                roll_amount,
                                axis=0)

            data.loc[:, cols_data] = app_data_np
            label.loc[:, cols_label] = label_np

            # print(label)

        return data, label


    def _reduce_labels(self, label: pd.DataFrame):
        columns_individual_labels = (label.columns != 'label')

        # print(columns_individual_labels)

        label.loc[:, columns_individual_labels] =\
            label.loc[:, columns_individual_labels].fillna(0).astype(np.uint8)

        label['label'] =\
            label.loc[:, columns_individual_labels]\
                .agg(lambda row: np.any(row).astype(np.uint8), axis=1)
        
        label = label.drop(label.columns[columns_individual_labels], axis=1)

        return label


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


    def _iaaft(self,
                data: np.array,
                sliding_window_size: int = 32,
                tolerance_percent: float = 5,
                candidate_count: int = 10,
                gradient_threshold: float = 2,
                gradient_mask_patience: int = 10) -> np.array:

        usable_processors = len(os.sched_getaffinity(0))
        process_pool = Pool(usable_processors//2)

        data_list_in = np.hsplit(data, data.shape[1])

        _apply_iaaft_with_args =\
                partial(_apply_iaaft,
                            sliding_window_size=sliding_window_size,
                            tolerance_percent=tolerance_percent,
                            candidate_count=candidate_count,
                            gradient_threshold=gradient_threshold,
                            gradient_mask_patience=gradient_mask_patience)
                            

        data_list_out = process_pool.map(_apply_iaaft_with_args,
                                                    data_list_in)

        process_pool.close()

        data = np.hstack(data_list_out)
            
        return data


    def fit_transform(self,
                        data: pd.DataFrame) -> pd.DataFrame:

        segments_augmented = []

        for augmentation, parameters in self.applied_augmentations:

            data_unmodified = data.copy()

            if len(segments_augmented) != 0:
                    data_unmodified.index =\
                        self._adjust_index(segments_augmented[-1].index,
                                                    data_unmodified.index)

            if augmentation == 'Identity':
                segments_augmented.append(data_unmodified)

            else:

                if augmentation == 'Roll':
                    data_augmented = self._roll_data(data_unmodified,
                                                parameters[0],
                                                parameters[1])

                    segments_augmented.append(data_augmented)
                    continue

                # If the augmentation starts with 'Roll',
                # but also contains other augmentations to
                # apply to the dataset, apply the roll and
                # then adapt the augmentation string and
                # parameter tuple to conform with what is
                # expected in the augmentations to apply 
                # afterwards

                elif augmentation.startswith('Roll'):
                    data_rolled_pd =\
                        self._roll_data(data_unmodified,
                                            parameters[0],
                                            parameters[1])

                    augmentation = '_'.join(augmentation.split('_')[1:])
                    parameters = parameters[2:]

                    segment_augmented_pd = data_rolled_pd

                else:
                    segment_augmented_pd = data_unmodified

                excluded_cols = segment_augmented_pd.columns\
                                    .str.startswith(_ordinal_cols_eclipse)

                data_to_augment_pd =\
                    segment_augmented_pd.loc[:, ~(excluded_cols)]

                data_to_augment_np = data_to_augment_pd.to_numpy()

                if augmentation == 'Scale':
                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    segment_augmented_np = data_to_augment_np*\
                                                    scaling_factor

                elif augmentation == 'APP':

                    segment_length_ratio = parameters[0]
                    amp_perturbation_amount = parameters[1]
                    phase_perturbation_amount = parameters[2]

                    segment_augmented_np =\
                        self._amplitude_phase_perturbation(data_to_augment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                elif augmentation == 'Scale_APP':

                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    data_to_augment_np = data_to_augment_np*\
                                                scaling_factor

                    segment_length_ratio = parameters[2]
                    amp_perturbation_amount = parameters[3]
                    phase_perturbation_amount = parameters[4]

                    segment_augmented_np =\
                        self._amplitude_phase_perturbation(data_to_augment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                data_augmented_pd = pd.DataFrame(segment_augmented_np,
                                                    data_to_augment_pd.index,
                                                    columns=data_to_augment_pd.columns)

                segment_augmented_pd.loc[:, ~excluded_cols] =\
                                                data_augmented_pd
                    
                segments_augmented.append(segment_augmented_pd)

        dataset_augmented = pd.concat(segments_augmented)

        fig, axes = plt.subplots(4, 1, figsize=(10, 24), dpi=300)

        fig.suptitle('Eclipse MemInfo Data')

        meminfo_data =\
            dataset_augmented.loc[:, dataset_augmented.columns.str.contains('meminfo')]

        for ax, app in zip(axes, self.applications):

            ax.set_title(app.upper())
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Data')

            ax.grid()

            meminfo_cols =  meminfo_data.columns.str.startswith(f'm_{app}_')

            data = meminfo_data.loc[:, meminfo_cols]

            if data.shape[-1]:

                data = data.fillna(0).to_numpy()

                data = MinMaxScaler().fit_transform(data)
                
                index = dataset_augmented.index.values

                ax.plot(index, data)
            
        plt.savefig('eclipse_augmentation_test_unlabeled.png')

        return dataset_augmented


    def fit_transform_labeled(self,
                                data: pd.DataFrame,
                                label: pd.DataFrame) -> pd.DataFrame:

        data_segments_augmented = []
        label_segments_augmented = []

        for augmentation, parameters in self.applied_augmentations:

            data_unmodified = data.copy()
            label_unmodified = label.copy()

            if len(data_segments_augmented) != 0:
                    data_unmodified.index =\
                        self._adjust_index(data_segments_augmented[-1].index,
                                                        data_unmodified.index)
                    label_unmodified.index =\
                        self._adjust_index(label_segments_augmented[-1].index,
                                                        label_unmodified.index)

            if augmentation == 'Identity':
                data_segments_augmented.append(data_unmodified)

                label_unmodified =\
                    self._reduce_labels(label_unmodified)

                label_segments_augmented.append(label_unmodified)

            else:

                if augmentation == 'Roll':
                    data_augmented, label_augmented =\
                            self._roll_data_labeled(data_unmodified,
                                                    label_unmodified,
                                                    parameters[0],
                                                    parameters[1])

                    label_augmented =\
                        self._reduce_labels(label_augmented)

                    data_segments_augmented.append(data_augmented)
                    label_segments_augmented.append(label_augmented)

                    continue

                # If the augmentation starts with 'Roll',
                # but also contains other augmentations to
                # apply to the dataset, apply the roll and
                # then adapt the augmentation string and
                # parameter tuple to conform with what is
                # expected in the augmentations to apply 
                # afterwards

                elif augmentation.startswith('Roll') and not\
                                        augmentation == 'Roll':
                    data_rolled_pd, label_rolled_pd =\
                        self._roll_data_labeled(data_unmodified,
                                                    label_unmodified,
                                                    parameters[0],
                                                    parameters[1])

                    label_rolled_pd =\
                        self._reduce_labels(label_rolled_pd)

                    label_segments_augmented.append(label_rolled_pd)


                    augmentation = '_'.join(augmentation.split('_')[1:])
                    parameters = parameters[2:]

                    data_segment_augmented_pd = data_rolled_pd

                else:

                    data_segment_augmented_pd = data_unmodified

                    label_segments_augmented.append(label_unmodified)

                excluded_cols = data_segment_augmented_pd.columns\
                                    .str.startswith(_ordinal_cols_eclipse)

                data_to_augment_pd =\
                    data_segment_augmented_pd.loc[:, ~(excluded_cols)]

                data_to_augment_np = data_to_augment_pd.to_numpy()

                if augmentation == 'Scale':
                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    segment_augmented_np = data_to_augment_np*\
                                                    scaling_factor

                elif augmentation == 'APP':

                    segment_length_ratio = parameters[0]
                    amp_perturbation_amount = parameters[1]
                    phase_perturbation_amount = parameters[2]

                    segment_augmented_np =\
                        self._amplitude_phase_perturbation(data_to_augment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                elif augmentation == 'Scale_APP':

                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    data_to_augment_np = data_to_augment_np*\
                                                scaling_factor

                    segment_length_ratio = parameters[2]
                    amp_perturbation_amount = parameters[3]
                    phase_perturbation_amount = parameters[4]

                    segment_augmented_np =\
                        self._amplitude_phase_perturbation(data_to_augment_np,
                                                            segment_length_ratio,
                                                            amp_perturbation_amount,
                                                            phase_perturbation_amount)

                data_augmented_pd = pd.DataFrame(segment_augmented_np,
                                                    data_to_augment_pd.index,
                                                    columns=data_to_augment_pd.columns)

                data_segment_augmented_pd.loc[:, ~excluded_cols] =\
                                                    data_augmented_pd
                    
                data_segments_augmented.append(data_segment_augmented_pd)

        dataset_augmented = pd.concat(data_segments_augmented)
        labels_augmented = pd.concat(label_segments_augmented)

        fig, axes = plt.subplots(4, 1, figsize=(10, 24), dpi=300)

        fig.suptitle('Eclipse MemInfo Data')

        meminfo_data =\
            dataset_augmented.loc[:, dataset_augmented.columns.str.contains('meminfo')]

        for ax, app in zip(axes, self.applications):

            ax.set_title(app.upper())
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Data')

            ax.grid()

            meminfo_cols =  meminfo_data.columns.str.startswith(f'm_{app}_')

            data = meminfo_data.loc[:, meminfo_cols]

            if data.shape[-1]:

                data = data.fillna(0).to_numpy()

                data = MinMaxScaler().fit_transform(data)
                
                index = dataset_augmented.index.values

                ax.plot(index, data)

                anomaly_starts, anomaly_ends =\
                    _get_runs_of_true(labels_augmented.to_numpy().flatten())
                    
                for start, end in zip(anomaly_starts, anomaly_ends):

                    start = max(0, start)
                    end = min(end, (len(index) - 1))

                    ax.axvspan(index[start], index[end], color='red', alpha=0.5)
            
        plt.savefig('eclipse_augmentation_test_labeled.png')

        return dataset_augmented, labels_augmented
