import os
import time
import warnings
from functools import partialmethod

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_loaders.omni_anomaly_dataset import OmniAnomalyDataset
from dataset_loaders.hlt_datasets import HLTDataset
from exp.exp_basic import ExpBasic
from models.model import Informer
from models.sad_like_loss import *
from utils.tools import EarlyStopping, adjust_learning_rate


def log_gradients_in_model(model, summary_writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            summary_writer.add_histogram(tag, value.cpu(), step)
            summary_writer.add_histogram(tag + "/grad", value.grad.cpu(), step)


class ExpInformer(ExpBasic):
    def __init__(self, args):
        super(ExpInformer, self).__init__(args)

    def _build_model(self):
        model = Informer(
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.no_distil,
            self.args.no_mix,
            self.device).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'machine-1-1': OmniAnomalyDataset,
            'HLT': HLTDataset,}

        Data = data_dict[self.args.data]

        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size

        elif flag=='pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1

        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        
        freq = args.freq

        dataset = None

        if Data == OmniAnomalyDataset:
            dataset = Data(dataset=self.args.data,
                            mode=flag,
                            size=[args.seq_len,
                                    args.label_len,
                                    args.pred_len],
                            features=args.features,
                            target=args.target,
                            inverse=args.inverse,
                            timeenc=timeenc,
                            freq=freq,
                            scaling_type='minmax',
                            scaling_source='train_set_fit')

        elif Data == HLTDataset:
            dataset = Data(mode=flag,
                            size=[args.seq_len,
                                    args.label_len,
                                    args.pred_len],
                            features=args.features,
                            target=args.target,
                            inverse=args.inverse,
                            timeenc=timeenc,
                            freq=freq,
                            scaling_type='minmax',
                            scaling_source='train_set_fit',
                            applied_augmentations=\
                                    self.args.augmentations,
                            augmented_dataset_size_relative=\
                                    self.args.augmented_dataset_size_relative,
                            augmented_data_ratio=\
                                    self.args.augmented_data_ratio)

        print(f'{flag} size: {len(dataset)}')

        data_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle_flag,
                                    num_workers=args.num_workers,
                                    drop_last=drop_last)

        return dataset, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim   


    def _select_criterion(self, loss='MSE'):

        if loss == 'MSE':
            return nn.MSELoss()

        elif loss == 'SMSE':
            return SADLikeLoss(eta=0.1)


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for batch_x,batch_y,batch_x_mark,batch_y_mark in tqdm(vali_loader):
            if self.args.output_attention:
                pred, true, _ = self._process_one_batch(vali_data,
                                                            batch_x,
                                                            batch_y,
                                                            batch_x_mark,
                                                            batch_y_mark)
            else:
                pred, true = self._process_one_batch(vali_data,
                                                        batch_x,
                                                        batch_y,
                                                        batch_x_mark,
                                                        batch_y_mark)


            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data.pickle_scaler(f'{path}/scaler.pkl')

        if self.args.loss == 'SMSE':
            labeled_train_data, labeled_train_loader =\
                            self._get_data(flag='labeled_train')

        train_steps_unlabeled = len(train_loader)

        train_steps_labeled = len(labeled_train_loader)\
                    if self.args.loss == 'SMSE' else 0

        delta = -1 if self.args.loss == 'SMSE' else 0

        early_stopping = EarlyStopping(patience=self.args.patience,
                                                        verbose=True,
                                                        delta=delta)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        summary_writer = SummaryWriter()

        for epoch in range(self.args.train_epochs):

            train_loss = []
            preds_all = []
            y_actual_all = []
            
            self.model.train()
            
            epoch_time = time.time()
            
            if self.args.loss == 'SMSE':
                for batch_index, (batch_x,\
                                    batch_y,\
                                    batch_x_mark,\
                                    batch_y_mark) in enumerate(tqdm(train_loader)):
                    
                    model_optim.zero_grad()

                    if self.args.output_attention:
                        pred, true, _ = self._process_one_batch(train_data,
                                                                    batch_x,
                                                                    batch_y,
                                                                    batch_x_mark,
                                                                    batch_y_mark)

                    else:
                        pred, true = self._process_one_batch(train_data,
                                                                batch_x,
                                                                batch_y,
                                                                batch_x_mark,
                                                                batch_y_mark)

                    preds_all.append(pred.detach().cpu().numpy())
                    y_actual_all.append(true.detach().cpu().numpy())

                    loss = criterion(pred, true)

                    train_loss.append(loss.item())
                     
                    summary_writer.add_scalar("Train loss",
                                                loss,
                                                batch_index +\
                                                    epoch*train_steps_unlabeled)

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                    # Stop training in semi-supervised setting
                    # when an amount of data equal to the
                    # unlabelled training set length minus the
                    # labelled training set length has been reached
                    # to ensure that the models receive the same
                    # amount of data

                    if batch_index >= (len(train_loader) -\
                                        len(labeled_train_loader)):
                        break

                for batch_index, (batch_x,\
                                    batch_y,\
                                    batch_x_mark,\
                                    batch_y_mark,\
                                    label) in enumerate(tqdm(labeled_train_loader)):

                    label = label.to(self.device)
                    
                    model_optim.zero_grad()

                    if self.args.output_attention:
                        pred, true, _ = self._process_one_batch(train_data,
                                                                    batch_x,
                                                                    batch_y,
                                                                    batch_x_mark,
                                                                    batch_y_mark)

                    else:
                        pred, true = self._process_one_batch(train_data,
                                                                batch_x,
                                                                batch_y,
                                                                batch_x_mark,
                                                                batch_y_mark)

                    preds_all.append(pred.detach().cpu().numpy())
                    y_actual_all.append(true.detach().cpu().numpy())

                    loss = criterion(pred, true, label)

                    train_loss.append(loss.item())

                    summary_writer.add_scalar("Train loss",
                                                loss, batch_index +\
                                                    train_steps_unlabeled -\
                                                    train_steps_labeled +\
                                                    epoch*train_steps_unlabeled)
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            else:

                for batch_index, (batch_x,\
                                    batch_y,\
                                    batch_x_mark,\
                                    batch_y_mark) in enumerate(tqdm(train_loader)):
                    
                    model_optim.zero_grad()

                    if self.args.output_attention:
                        pred, true, _ = self._process_one_batch(train_data,
                                                                    batch_x,
                                                                    batch_y,
                                                                    batch_x_mark,
                                                                    batch_y_mark)

                    else:
                        pred, true = self._process_one_batch(train_data,
                                                                batch_x,
                                                                batch_y,
                                                                batch_x_mark,
                                                                batch_y_mark)

                    preds_all.append(pred.detach().cpu().numpy())
                    y_actual_all.append(true.detach().cpu().numpy())

                    loss = criterion(pred, true)

                    train_loss.append(loss.item())
                    
                    summary_writer.add_scalar("Train loss",
                                                loss,
                                                batch_index + epoch*\
                                                    train_steps_unlabeled)
                     
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))    

            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data,
                                    vali_loader,
                                    criterion)

            preds_all = early_stopping(vali_loss,
                                        self.model,
                                        preds_all,
                                        path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            summary_writer.add_scalar("Validation loss", vali_loss, epoch)

            log_gradients_in_model(self.model,
                                    summary_writer,
                                    epoch)

        preds_all_np = np.array(preds_all)
        y_actual_all = np.array(y_actual_all)

        preds_all_np = preds_all_np.reshape(-1, preds_all_np.shape[-2], preds_all_np.shape[-1])
        y_actual_all = y_actual_all.reshape(-1, y_actual_all.shape[-2], y_actual_all.shape[-1])

        # Save results

        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds_all_train.npy', preds_all_np)
        np.save(folder_path + 'true_values_all_train.npy', y_actual_all)

        best_model_path = path +\
                '/checkpoint_informer.pth'

        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting):

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/checkpoint_informer.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds_all = []
        y_actual_all = []
        
        with tqdm(total=len(test_loader)) as pbar:
            for count, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                if self.args.output_attention:

                    pred, true, attention =\
                            self._process_one_batch(test_data,
                                                        batch_x,
                                                        batch_y,
                                                        batch_x_mark,
                                                        batch_y_mark)

                else:
                    pred, true = self._process_one_batch(test_data,
                                                            batch_x,
                                                            batch_y,
                                                            batch_x_mark,
                                                            batch_y_mark)

                
                preds_all.append(pred.detach().cpu().numpy())
                y_actual_all.append(true.detach().cpu().numpy())

                pbar.update(1)

        preds_all = np.array(preds_all)
        y_actual_all = np.array(y_actual_all)

        preds_all = preds_all.reshape(-1, preds_all.shape[-2], preds_all.shape[-1])
        y_actual_all = y_actual_all.reshape(-1, y_actual_all.shape[-2], y_actual_all.shape[-1])
        
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds_all_test.npy', preds_all)
        np.save(folder_path + 'true_values_all_test.npy', y_actual_all)
        np.save(folder_path + 'labels_all_test.npy', test_data.get_labels())


    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Decoder input

        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        # Encoder - decoder

        if self.args.use_amp:
            with torch.cuda.amp.autocast():

                if self.args.output_attention:
                    outputs, attention = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:

            if self.args.output_attention:
                outputs, attention = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


        if self.args.output_attention:
            return outputs, batch_y, attention
        else:
            return outputs, batch_y