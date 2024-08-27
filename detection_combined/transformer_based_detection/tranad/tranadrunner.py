import logging
import os
import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .src.models import TranAD, DAGMM, USAD, OmniAnomaly
from .src.dlutils import ComputeLoss
from .src.datapreprocessor import DataPreprocessor
from utils.anomalyclassification import AnomalyType
from utils.tqdmloggingdecorator import tqdmloggingdecorator

device='cuda:0'


class TranADRunner():
    def __init__(self,
                    checkpoint_dir,
                    nan_output_tolerance_period: int = 10,
                    model: str = 'TranAD',
                    variant: str = '2018') -> None:

        self.checkpoint_dir = checkpoint_dir
        self._nan_output_tolerance_period =\
                        nan_output_tolerance_period
        self._model = model
        self._variant = variant

        with open(self.checkpoint_dir +\
                    '/model_parameters.json', 'r') as parameter_dict_file:
            self.parameter_dict = json.load(parameter_dict_file)

        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0')
        
        self.data_preprocessor = DataPreprocessor(self.parameter_dict,
                                                    self.checkpoint_dir)
        
        feats = 102 if self._variant == '2018' else 104

        if self._model == 'TranAD':
            self.model = TranAD(feats).double().to(device)
        elif self._model == 'USAD':
            self.model = USAD(feats).double().to(device)
        elif self._model == 'OmniAnomaly':
            self.model = OmniAnomaly(feats).double().to(device)
        elif self._model == 'DAGMM':
            self.model = DAGMM(feats).double().to(device)
        else:
            raise ValueError(f'Model {self._model} unknown')

        fname = f'{self.checkpoint_dir}/model.ckpt'

        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self._predictions_all = []

        self.model.eval()

        self._logger = logging.getLogger(__name__)

        self._anomaly_start = None

        self._anomaly_duration = 0

        self._hidden = None


    def _convert_to_windows(self, data):
        return torch.stack([data.view(-1)])


    def backprop(self,
                    model,
                    data):

        feats = 102 if self._variant == '2018' else 104

        if self._model == 'DAGMM':
            l = nn.MSELoss(reduction = 'none')
            l1s = []; l2s = []

            data = data.to(device).contiguous()

            data = self._convert_to_windows(data)

            ae1s = []
            _, x_hat, _, _ = model(data)
            ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)

            torch.cuda.empty_cache()

            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()[0]

        elif self._model == 'OmniAnomaly':
            l = nn.MSELoss(reduction = 'none')
            y_preds = []

            data = data.to(device).contiguous()

            data = self._convert_to_windows(data)

            y_pred, _, _, self._hidden = model(data, self._hidden)
            y_preds.append(y_pred)

            torch.cuda.empty_cache()
        
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().cpu().numpy(), y_pred.detach().cpu().numpy()[0]


        elif  self._model == 'USAD':
            l = nn.MSELoss(reduction = 'none')
            l1s, l2s = [], []

            ae1s, ae2s, ae2ae1s = [], [], []

            data = data.to(device).contiguous()

            data = self._convert_to_windows(data)

            ae1, ae2, ae2ae1 = model(data)
            ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)

            torch.cuda.empty_cache()

            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1*l(ae1s, data) + 0.9*l(ae2ae1s, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()[0]

        else:

            l = nn.MSELoss(reduction = 'none')
            data_x = data
            
            zs = []
            elems = []

            data_x = data_x.to(device)

            window = data_x.permute(1, 0, 2)

            elem = window[-1, :, :].view(1, 1, feats)

            elems.append(elem.detach().cpu())

            z = model(window, elem)

            if isinstance(z, tuple): z = z[1]

            zs.append(z.detach().cpu())

            torch.cuda.empty_cache()

            z = torch.cat(zs, dim=0)
            elem = torch.cat(elems, dim=0)

            z = torch.reshape(z, (1, z.shape[0]*z.shape[1], -1))
            elem = torch.reshape(elem, (1, elem.shape[0]*elem.shape[1], -1))

            loss = l(z, elem)[0]

            return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]

    
    @tqdmloggingdecorator
    def detect(self, data: pd.DataFrame):

        data_x = self.data_preprocessor.process(data)

        timestamp = data.index[-1]

        loss, y_pred = self.backprop(self.model, data_x)

        lossFinal = np.mean(loss, axis=1)

        for loss in lossFinal:

            # Note: This is currently not using proper SPOT thresholding.
            # The generated logs for transformer-based detection are not
            # used for evaluation. Instead, we directly use the generated
            # predictions.

            if loss > 0.5:

                if self._anomaly_duration == 0:
                    self._anomaly_start =\
                            timestamp.strftime('%Y-%m-%d %H:%M:%S')

                    self._logger.info('Transformer-based detection '
                                        'encountered anomaly at timestamp '
                                        f'{self._anomaly_start}')

                self._anomaly_duration += 1

                self.detection_callback(0, AnomalyType.TransformerBased,
                                            self._anomaly_start,
                                            self._anomaly_duration)

            else:
                self._anomaly_duration = 0

            self._predictions_all.append(loss)


    def register_detection_callback(self,
                                        callback: Callable) -> None:
        self.detection_callback = callback


    def get_predictions(self) -> np.array:
        return np.array(self._predictions_all)
