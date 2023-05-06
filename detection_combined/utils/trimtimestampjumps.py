import numpy as np
import pandas as pd
import logging

def trim_timestamp_jumps(data: pd.DataFrame,
                            frequency_expected_seconds: float = 5,
                            tolerance_seconds: float = 2) -> pd.DataFrame:

        logger = logging.getLogger(trim_timestamp_jumps.__name__)

        index = data.index

        delta = index[1:] - index[:-1]

        index = pd.Series(index)

        threshold_lower = pd.Timedelta(
                            frequency_expected_seconds -\
                                            tolerance_seconds, unit='s')

        threshold_upper = pd.Timedelta(
                            frequency_expected_seconds +\
                                            tolerance_seconds, unit='s')

        timestamp_jump_indices =\
                np.union1d(np.flatnonzero(delta <\
                                            threshold_lower),
                                np.flatnonzero(delta >\
                                                threshold_upper))

        if len(timestamp_jump_indices):
            logger.info(f'Encountered {len(timestamp_jump_indices)} '
                                'timestamp jump(s), trimming dataset')

            logger.info(f'Dataset length before trim: {len(data)}')

            if len(timestamp_jump_indices) == 1:

                closer_to_beginning =\
                    (timestamp_jump_indices[0] < len(data)//2)

                if closer_to_beginning:

                    logger.info('Timestamp jump is at index '
                                f'{timestamp_jump_indices[0]}, '
                                'closer to beginning of dataset. '
                                'Trimming dataset until timestamp jump')

                    data = data.iloc[:timestamp_jump_indices[0], :]
                else:

                    logger.info('Timestamp jump is at index '
                                f'{timestamp_jump_indices[0]}, '
                                'closer to end of dataset. '
                                'Trimming dataset from timestamp jump on')

                    data = data.iloc[timestamp_jump_indices[0]:, :]

            # elif len(timestamp_jump_indices) == 2:

                # logger.info('Trimming dataset between indices '
                #                 f'{timestamp_jump_indices[0]} and '
                #                 f'{timestamp_jump_indices[1]}')

                # data = data.iloc[timestamp_jump_indices[0]:timestamp_jump_indices[1], :]

            else:
                logger.info('Trimming after last jump at index '
                                    f'{timestamp_jump_indices[-1]}')

                data = data.iloc[timestamp_jump_indices[-1]:, :]

            logger.info(f'Dataset length after trim: {len(data)}')

            if not data.index.is_monotonic_increasing:
                error_string = 'Dataset index after trimming is not '\
                                    'strictly monotonically increasing'

                logger.error(error_string)
                raise RuntimeError(error_string)

            timestamp_jump_indices =\
                    np.union1d(np.flatnonzero(delta <\
                                                threshold_lower),
                                    np.flatnonzero(delta >\
                                                    threshold_upper))

            if len(timestamp_jump_indices):
                error_string = 'Data retains timestamp jumps'

                logger.error(error_string)
                raise RuntimeError(error_string)

        return data