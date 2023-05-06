import numpy as np
import pandas as pd
import torch
import pickle as pkl
import logging

from utils.variables import nan_fill_value

class DataPreprocessor():
    def __init__(self,
                    parameter_dict: dict,
                    checkpoint_dir: str,
                    data_clipping_val: float = 1e9) -> None:

        self.window_size = parameter_dict['window_size']
        
        scaler_dir_and_filename =\
                    checkpoint_dir +\
                    '/scaler.pkl'

        self.scaler = pkl.load(open(
                        scaler_dir_and_filename, 'rb'))

        self._data_clipping_val =\
                    data_clipping_val

        self._logger = logging.getLogger(__name__)


    def process(self, data: pd.DataFrame):

        data_x = data.to_numpy()

        if np.any(data_x == nan_fill_value):

            _, data_x_nan_indices =\
                np.nonzero(data_x == nan_fill_value)

            missing_subgroups =\
                [subgroup.removeprefix('m_')\
                    for subgroup in data.columns[data_x_nan_indices]]

            missing_subgroups = list(dict.fromkeys(missing_subgroups))

            if len(missing_subgroups) == 1:
                missing_subgroups_formatted = missing_subgroups[0]
            else:
                missing_subgroups_formatted =\
                        ', '.join(missing_subgroups)

            warning_string =\
                'Half or more of the elements in subgroup(s) '\
                f'{missing_subgroups_formatted} '\
                'are inactive. Second stage '\
                'detection might be affected'

            self._logger.warning(warning_string)

        data_x_scaled = self.scaler.transform(data_x)

        data_x_scaled = torch.from_numpy(data_x_scaled)


        if data_x_scaled.dim() == 2:
            data_x_scaled = torch.unsqueeze(data_x_scaled, 0)

        return data_x_scaled


    def register_data_provider_fn(self,
                                    data_provider_fn) -> None:
        self.data_provider_fn =\
                        data_provider_fn


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def get_sequence_length(self):
        return self.seq_len