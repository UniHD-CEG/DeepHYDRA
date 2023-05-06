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

from utils.iaaft import surrogates

_implemented_augmentations = ['Scale',
                                'Dilate',
                                'APP',
                                'Scale_APP',
                                'IAAFT',
                                'Scale_IAAFT']


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


def _plot_colourline(x, y, c):

    c = cm.Dark2((c - np.min(c))/(np.max(c) - np.min(c)))

    ax = plt.gca()
    
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i])
    return


def _get_gradient_based_mask(data: np.array,
                                gradient_threshold: float = 5,
                                patience: int = 10) -> np.array:

    gradient_abs = np.abs(np.gradient(data, axis=0))
    
    mask = np.where(gradient_abs >= gradient_threshold, True, False)

    # Mask time periods between POTs of length
    # equal to or greater than the patience period

    if mask.ndim == 1:
        pots = np.flatnonzero(mask)
        for row_count in range(len(pots) - 1):
            if pots[row_count + 1] -\
                        pots[row_count] <= patience:
                mask[pots[row_count]:pots[row_count + 1]] = True

    elif mask.ndim == 2:
        for column in range(mask.shape[1]):
            pots = np.flatnonzero(mask[:, column])
            for row_count in range(len(pots) - 1):
                if pots[row_count + 1] -\
                            pots[row_count] <= patience:
                    mask[pots[row_count]:pots[row_count + 1], column] = True
    else:
        raise ValueError('_get_gradient_based_mask()'
                            'only supports 1d or 2d arrays')

    mask = np.invert(mask)

    return mask


def _apply_iaaft(data: np.array,
                    sliding_window_size: int = 10,
                    tolerance_percent: float = 5,
                    candidate_count: int = 10,
                    gradient_threshold: float = 5,
                    gradient_mask_patience: int = 10) -> np.array:

    # stride = 3*sliding_window_size//4
    stride = sliding_window_size

    gradient_based_mask =\
        _get_gradient_based_mask(data,
                                    gradient_threshold,
                                    gradient_mask_patience)
        
    segment_starts, segment_lengths =\
                _get_runs_of_true(gradient_based_mask.flatten())

    for segment_start, segment_length in\
                zip(segment_starts, segment_lengths):

        segment_end = segment_start + segment_length

        segment = data[segment_start:segment_end, 0]

        if segment_length > sliding_window_size:

            sliding_window =\
                    np_st.sliding_window_view(segment,
                                                sliding_window_size,
                                                writeable=True)[::stride, :]

            for window in range(len(sliding_window)):

                surrogates_windows =\
                            surrogates(sliding_window[window, :],
                                                    candidate_count,
                                                    tolerance_percent,
                                                    verbose=False)

                # mse_min = np.finfo('d').max
                # index_mse_min = 0

                # for index, surrogate in enumerate(surrogates_windows):

                #     mse = mean_squared_error(
                #                 sliding_window[window, :],
                #                 surrogate)

                #     # print(f'Index: {index}\tMSE: {mse:.3f}')
                            
                #     if mse < mse_min:
                #         index_mse_min = index
                #         mse_min = mse

                # print(f'Choosing surrogate {index_mse_min} '
                #                 f'with MSE of {mse_min:.3f}')

                # sliding_window[window, :] =\
                #             surrogates_windows[index_mse_min]

                sliding_window[window, :] =\
                             surrogates_windows[0]

            else:
                surrogate_segments = surrogates(segment,
                                                candidate_count,
                                                tolerance_percent,
                                                verbose=False)

                # mse_min = np.finfo('d').max
                # index_mse_min = 0

                # for index, surrogate in enumerate(surrogate_segments):

                #     mse = mean_squared_error(segment,
                #                                 surrogate)

                #     # print(f'Index: {index}\tMSE: {mse:.3f}')
                            
                #     if mse < mse_min:
                #         index_mse_min = index
                #         mse_min = mse

                # # print(f'Choosing surrogate {index_mse_min} '
                # #                 f'with MSE of {mse_min:.3f}')

                # segment = surrogate_segments[index_mse_min]

                segment = surrogate_segments[0]

    return data
    

class ATLASDataTimeseriesAugmentor():

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

            elif augmentation == 'Dilate':
                pass

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

            elif augmentation == 'IAAFT':
                pass

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

        # print(collision_window_ends)

#         colormap = np.empty(len(data))
# 
#         window_end_index = 0
# 
#         for index in range(len(data)):
#             colormap[index] = window_end_index
# 
#             if window_end_index < len(collision_window_ends):
#                 if index == (collision_window_ends[window_end_index] - 1):
#                     window_end_index += 1
# 
#         fig, ax = plt.subplots(figsize=(10, 5))
# 
#         x = np.arange(len(data))
#         # x = np.arange(len(data[11000:11220]))
# 
#         plt.grid()
# 
#         # _plot_colourline(x, data[:, 0], colormap)
#         _plot_colourline(x, data[:, 51], colormap)
#         # _plot_colourline(x, data[11000:11220, 0], colormap[11000:11220])
# 
#         plt.tight_layout()
#         
#         # plt.savefig(f'l1_dataset_segmentation_test.png', dpi=300)
#         plt.savefig(f'l1_dataset_segmentation_test_stdev.png', dpi=300)
#         # plt.savefig(f'l1_dataset_segmentation_test_cropped.png', dpi=300)

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


    def _adjust_index(self,
                        index_previous: pd.DatetimeIndex,
                        index_current: pd.DatetimeIndex) -> pd.DatetimeIndex:
        
        index_current = pd.Series(index_current)

        # print(f'Last timestamp previous segment: {index_previous[-1]}')
        # print(f'Original first timestamp current segment: {index_current[0]}')

        index_current -= index_current[0] - index_previous[-1]
        index_current += pd.Timedelta(5, unit='s')

        # print(f'Adjusted first timestamp current segment: {index_current[0]}')

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
                pd.Series(data.index[:-1] -data.index[1:]).std()

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

        # fig, (ax_original, ax_augmented) =\
        #             plt.subplots(2, 1, figsize=(10, 8))
        #
        # x = np.arange(len(data))

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

                # gaussian_noise_amp =\
                #         self.rng.normal(amp_data_window_mean,
                #                             amp_data_window_std,
                #                             size=(segment_length,))

                # sliding_window_amp_data[window, :] =\
                #                             amp_perturbation_amount*\
                #                             gaussian_noise_amp

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

            # ax_original.plot(x, data[:, column])

            data[:, column] =\
                np.maximum(0, ifft(amp_data*e_i_phase))
    
            # ax_augmented.plot(x, data[:, column])

        return data

        # ax_original.grid(True)
        # ax_augmented.grid(True)

        # ax_original.set_title('Original Data')
        # ax_original.set_xlabel('Timestep')
        # ax_original.set_ylabel('L1 Rate')

        # ax_augmented.set_title('Augmented Data')
        # ax_augmented.set_xlabel('Timestep')
        # ax_augmented.set_ylabel('L1 Rate')

        # plt.tight_layout()
        
        # plt.savefig('app_test.png', dpi=300)

        # exit()

    def _iaaft(self,
                data: np.array,
                sliding_window_size: int = 32,
                tolerance_percent: float = 5,
                candidate_count: int = 10,
                gradient_threshold: float = 2,
                gradient_mask_patience: int = 10) -> np.array:

        fig, (ax_original, ax_augmented) =\
                    plt.subplots(2, 1, figsize=(10, 8))

        x = np.arange(len(data))

        ax_original.plot(x, data)
        # ax_original.plot(x[300:450], data[300:450])
        # ax_original.plot(x[750:1000], data[750:1000, :51])

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

        # ax_augmented.plot(x, data)
        # # ax_augmented.plot(x[300:450], data[300:450])
        # # ax_augmented.plot(x[750:1000], data[750:1000, :51])

        # ax_original.grid(True)
        # ax_augmented.grid(True)

        # ax_original.set_title('Original Data')
        # ax_original.set_xlabel('Timestep')
        # ax_original.set_ylabel('L1 Rate')

        # ax_augmented.set_title('Augmented Data')
        # ax_augmented.set_xlabel('Timestep')
        # ax_augmented.set_ylabel('L1 Rate')

        # plt.tight_layout()
        
        # plt.savefig('iaaft_test.png', dpi=300)

        # fig, (ax_data, ax_grad, ax_grad_mask) =\
        #             plt.subplots(3, 1, figsize=(10, 8))

        # x = np.arange(len(data))

        # ax_data.plot(x, data)

        # grad = np.gradient(data, axis=0)
        # ax_grad.plot(x, grad)

        # gradient_based_mask =\
        #         _get_gradient_based_mask(data)
        
        # ax_grad_mask.plot(x, gradient_based_mask)

        # ax_data.grid(True)
        # ax_grad.grid(True)
        # ax_grad_mask.grid(True)

        # ax_data.set_title('Data')
        # ax_data.set_xlabel('Timestep')
        # ax_data.set_ylabel('L1 Rate')

        # ax_grad.set_title('First Derivative')
        # ax_grad.set_xlabel('Timestep')
        # ax_grad.set_ylabel('First Derivative')

        # ax_grad_mask.set_title('Gradient-Based Mask')
        # ax_grad_mask.set_xlabel('Timestep')
        # ax_grad_mask.set_ylabel('Gradient-Based Mask')

        # plt.tight_layout()
        
        # plt.savefig('iaaft_test.png', dpi=300)

        # exit()
            
        return data


    def fit_transform(self,
                        data: pd.DataFrame,
                        target_size_min: int,
                        ratio_augmented: float) -> pd.DataFrame:

        segments_unmodified = self._segment_dataframe(data, 5, 0.5)

        segments_augmented = []
        augmented_dataset_size = 0

        colormap = []

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

                colormap.append(np.full(len(source_segment),
                                        index_source_segment))

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

                elif augmentation == 'Dilate':
                    pass

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


                elif augmentation == 'IAAFT':

                    sliding_window_size = parameters[0]
                    tolerance_percent = parameters[1]
                    candidate_count = parameters[2]
                    gradient_threshold = parameters[3]
                    gradient_mask_patience = parameters[4]

                    segment_augmented_np =\
                            self._iaaft(source_segment_np,
                                            sliding_window_size,
                                            tolerance_percent,
                                            candidate_count,
                                            gradient_threshold,
                                            gradient_mask_patience)

                elif augmentation == 'Scale_IAAFT':

                    scaling_factor =\
                        self.rng.uniform(parameters[0], parameters[1])

                    source_segment_np = source_segment_np*\
                                                scaling_factor

                    sliding_window_size = parameters[2]
                    tolerance_percent = parameters[3]
                    candidate_count = parameters[4]
                    gradient_threshold = parameters[5]
                    gradient_mask_patience = parameters[6]

                    segment_augmented_np =\
                            self._iaaft(source_segment_np,
                                            sliding_window_size,
                                            tolerance_percent,
                                            candidate_count,
                                            gradient_threshold,
                                            gradient_mask_patience)

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

                colormap.append(np.full(len(segment_augmented_pd),
                                                index_source_segment))

            augmented_dataset_size += len(source_segment)

        dataset_augmented = pd.concat(segments_augmented)

        # for count in range(1, len(dataset_augmented)):
        #     if dataset_augmented.index[count] <=\
        #                 dataset_augmented.index[count - 1]:
        #         print(f'Timestamps not monotic increasing '
        #                 f'between index {count - 1} and {count}')
        #         print(f'\tTimestamp at {count - 1}: '
        #                 f'{dataset_augmented.index[count - 1]}'
        #                 f'\n\tTimestamp at {count}: '
        #                 f'{dataset_augmented.index[count]}')

        # assert dataset_augmented.index.is_monotonic_increasing

        # colormap = np.concatenate(colormap)

        # fig, ax = plt.subplots(figsize=(10, 5))

        # x = np.arange(len(dataset_augmented))

        # plt.grid()

        # _plot_colourline(x, dataset_augmented.iloc[:, 0], colormap)

        # plt.tight_layout()
        
        # plt.savefig(f'l1_dataset_augmentation_test.png', dpi=300)

        # exit()


        return dataset_augmented