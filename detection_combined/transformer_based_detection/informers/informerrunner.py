import logging
import json
import pickle as pkl
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
# from torch import optim
# from torch.utils.data import DataLoader

from .models.model import Informer
from .utils.datapreprocessor import DataPreprocessor
import utils.spotunpickler as supkl
from utils.spot import SPOT
from utils.exceptions import NonCriticalPredictionException
from utils.anomalyclassification import AnomalyType
from utils.tqdmloggingdecorator import tqdmloggingdecorator

class InformerRunner():
    def __init__(self,
                    checkpoint_dir,
                    nan_output_tolerance_period: int = 10,
                    loss_type: str = 'mse',
                    use_spot_detection: bool = False,
                    device: str='cuda:0') -> None:

        self.checkpoint_dir = checkpoint_dir
        self._nan_output_tolerance_period =\
                        nan_output_tolerance_period
        self._loss_type = loss_type
        self._use_spot_detection = use_spot_detection

        with open(self.checkpoint_dir +\
                    '/model_parameters.json', 'r') as parameter_dict_file:
            self.parameter_dict = json.load(parameter_dict_file)

        self.device = torch.device(device)

        self.parameter_dict['timeenc'] = 0\
                    if self.parameter_dict['embed'] != 'timeF' else 1
        
        self.data_preprocessor = DataPreprocessor(self.parameter_dict,
                                                    self.checkpoint_dir)
        
        self.model = self.load_model().to(self.device)

        path = self.checkpoint_dir

        model_path = path + '/checkpoint_informer.pth'
        self.model.load_state_dict(torch.load(model_path,
                                                map_location=self.device))

        self._predictions_all = []

        self.model.eval()

        self._logger = logging.getLogger(__name__)

        self._spot = None

        if self._use_spot_detection:
            spot_path = f'{path}/spot_informer_{self._loss_type}.pkl'
            self._spot = supkl.load(open(spot_path, 'rb'))
            
        self._data_x_last = None

        self._anomaly_start = None
        self._anomaly_duration = 0

        self._nan_output_count = 0
        

    def load_model(self):

        model = Informer(self.parameter_dict['enc_in'],
                                self.parameter_dict['dec_in'], 
                                self.parameter_dict['c_out'], 
                                self.parameter_dict['seq_len'], 
                                self.parameter_dict['label_len'],
                                self.parameter_dict['pred_len'], 
                                self.parameter_dict['factor'],
                                self.parameter_dict['d_model'], 
                                self.parameter_dict['n_heads'], 
                                self.parameter_dict['e_layers'],
                                self.parameter_dict['d_layers'], 
                                self.parameter_dict['d_ff'],
                                self.parameter_dict['dropout'], 
                                self.parameter_dict['attn'],
                                self.parameter_dict['embed'],
                                self.parameter_dict['freq'],
                                self.parameter_dict['activation'],
                                True,
                                self.parameter_dict['distil'],
                                self.parameter_dict['mix'],
                                self.device).float()

        return model

    
    @tqdmloggingdecorator
    def detect(self, data: pd.DataFrame):

        data_x,\
            data_y,\
            data_x_mark,\
            data_y_mark = self.data_preprocessor.process(data)

        timestamp = data.index[-1]

        viz_data = data.to_numpy()[:self.parameter_dict['seq_len'], :]

        preds, _ = self._process_one_batch(self.data_preprocessor,
                                                            data_x,
                                                            data_y,
                                                            data_x_mark,
                                                            data_y_mark,
                                                            viz_data)
                
        preds = preds.detach().cpu().numpy()

        if np.any(np.isnan(preds)):
            self._nan_output_count += 1

            if self._nan_output_count >=\
                    self._nan_output_tolerance_period:

                self._logger.warning('Encountered NaN in '
                                        'Informer predictions')
                self._logger.error(f'Reached threshold of tolerated '
                                    'consecutive NaN predictions of '
                                    f'{self._nan_output_tolerance_period}')
                
                raise NonCriticalPredictionException(
                                    f'Informer reached threshold of tolerated '
                                    'consecutive NaN predictions of '
                                    f'{self._nan_output_tolerance_period}')

            else:
                self._logger.warning('Encountered NaN in '
                                        'Informer predictions, skipping '
                                        'kNN anomaly prediction step')
                self._logger.warning(f'Consecutive NaN predictions:'
                                            f'{self._nan_output_count} '
                                            'tolerated NaN predictions: '
                                            f'{self._nan_output_tolerance_period}')
            
            return

        else:
            self._nan_output_count = 0

        if not isinstance(self._data_x_last, type(None)):

            l2_dist =\
                np.mean((preds[:, 0, :] - self._data_x_last[:, -1, :])**2, 1)[0]

            self._predictions_all.append(l2_dist)
        
            if self._use_spot_detection:
                l2_dist_detection = self._spot.run_online([l2_dist])
            else:
                l2_dist_detection = (l2_dist > 0.5)

            if l2_dist_detection:

                if self._anomaly_duration == 0:
                    self._anomaly_start =\
                            timestamp.strftime('%Y-%m-%d %H:%M:%S')

                    self._logger.info('Transformer-based detection '
                                        'encountered anomaly at timestamp '
                                        f'{self._anomaly_start} '
                                        'using L2 dist')

                self._anomaly_duration += 1

            else:
                self._anomaly_duration = 0

        self._data_x_last = data_x.detach().cpu().numpy()


    def _process_one_batch(self,
                            dataset_object,
                            batch_x,
                            batch_y,
                            batch_x_mark,
                            batch_y_mark,
                            viz_data):

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Decoder input

        padding = self.parameter_dict['padding']
        pred_len = self.parameter_dict['pred_len']
        label_len = self.parameter_dict['label_len']

        if padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()

        elif padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(self.device)

        # Encoder - decoder

        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                            viz_data=viz_data)

        if self.parameter_dict['inverse']:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = 0
        batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)

        return outputs, batch_y


    def register_detection_callback(self,
                                        callback: Callable) -> None:
        self.detection_callback = callback


    def get_predictions(self) -> np.array:
        return np.array(self._predictions_all)
