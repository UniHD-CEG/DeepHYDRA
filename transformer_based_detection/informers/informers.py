import argparse
import json

import torch
import numpy as np

from exp.exp_informer import ExpInformer

output_dir = '../../evaluation/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Informer-MSE/Informer-SMSE Anomaly Detection')

    parser.add_argument('--data', type=str, required=True, default='HLT', help='data')
    parser.add_argument('--seed', type=float, default=42, help='Random seed')
    
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=32, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=102, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=102, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=102, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--no_distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--no_mix', action='store_false', help='Don\'t mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='exp description')
    parser.add_argument('--loss', type=str, default='MSE',help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    parser.add_argument('--apply_augmentations', action='store_true', help='Apply augmentations to the training set')
    parser.add_argument('--augmentations', type=str, nargs='+', help='Applied augmentations and the factors used in augmentation')
    parser.add_argument('--augmented_dataset_size_relative', type=float, default=1., help='Size of the augmented training set relative to the original training set')
    parser.add_argument('--augmented_data_ratio', type=float, default=0., help='Amount of augmented data in the augmented training set')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    Exp = ExpInformer

    augmentations = []
    augmentation_string = ''

    if args.apply_augmentations:
        
        for augmentation in args.augmentations:
            augmentation = augmentation.replace(' ', '')
            aug_type, factors = augmentation.split(':')

            factors = factors.split(',')

            factors_string = '_'.join(factors)

            factors = [float(factor) for factor in factors]

            if len(augmentation_string):
                augmentation_string += '_'

            augmentation_string += aug_type + '_' +\
                                        factors_string

            augmentations.append((aug_type, factors))
        
        args.augmentations = augmentations

    augmentation_string += f'_rel_size_{args.augmented_dataset_size_relative}'\
                                        f'_ratio_{args.augmented_data_ratio:.2f}'

    setting = f'{args.loss.lower()}_{augmentation_string.lower()}'

    # Set experiments
    exp = Exp(args)
    
    exp.train(setting)
    exp.test(setting)

    # Load and preprocess train set results

    preds_all_train = np.load('./results/' + setting + '/preds_all_train.npy')
    preds_all_train = preds_all_train.squeeze()

    trues_all_train = np.load('./results/' + setting + '/true_values_all_train.npy')
    trues_all_train = trues_all_train.squeeze()

    l2_distances_all_train = np.mean((preds_all_train[:, :] - trues_all_train[:, :])**2, 1)

    subfolder = 'reduced_detection' if args.data == 'HLT' else 'smd'

    postfix = ''

    if args.apply_augmentations:
        if args.augmented_data_ratio != 0:
            postfix = ''
        else:
            postfix = '_no_augment'

    np.save(f'{output_dir}/{subfolder}/predictions/'
                f'l2_dist_train_{args.loss.lower()}{postfix}.npy',
                                                l2_distances_all_train)
    
    if args.data == 'HLT':
        np.save(f'{output_dir}/combined_detection/predictions/'
                    f'l2_dist_train_{args.loss.lower()}{postfix}.npy',
                                                l2_distances_all_train)

    # Load and preprocess test set results

    preds_all_test = np.load('./results/' + setting + '/preds_all_test.npy')
    preds_all_test = preds_all_test.squeeze()

    trues_all_test = np.load('./results/' + setting + '/true_values_all_test.npy')
    trues_all_test = trues_all_test.squeeze()

    l2_distances_all_test = np.mean((preds_all_test[:, :] - trues_all_test[:, :])**2, 1)

    np.save(f'{output_dir}/{subfolder}/'
                f'l2_dist_{args.loss.lower()}{postfix}.npy',
                                        l2_distances_all_test)
    
    parameter_dict = {"model": "informer",
                            "data": args.data,
                            "freq": args.freq,
                            "seq_len": args.seq_len,
                            "label_len": args.label_len,
                            "pred_len": args.pred_len,
                            "enc_in": args.enc_in,
                            "dec_in": args.dec_in,
                            "c_out": args.c_out,
                            "d_model": args.d_model,
                            "n_heads": args.n_heads,
                            "e_layers": args.e_layers,
                            "d_layers": args.d_layers,
                            "d_ff": args.d_ff,
                            "factor": args.factor,
                            "padding": args.padding,
                            "distil": args.no_distil,
                            "dropout": args.dropout,
                            "attn": args.attn,
                            "embed": args.embed,
                            "activation": args.activation,
                            "mix": args.no_mix,
                            "inverse": args.inverse}

    with open(f'{args.checkpoints}/{setting}/model_parameters.json', 'w') as parameter_dict_file:
        json.dump(parameter_dict,
                    parameter_dict_file)

