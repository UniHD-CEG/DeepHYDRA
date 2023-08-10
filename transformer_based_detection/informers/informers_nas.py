import os
from shutil import rmtree
from functools import partial, partialmethod
import argparse

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,\
                                precision_recall_fscore_support,\
                                matthews_corrcoef
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.spot import SPOT
from exp.exp_informer import ExpInformer

output_dir = '../../evaluation'


def get_anomalous_runs(x):
    '''
    Find runs of consecutive items in an array.
    As published in https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    '''

    # Ensure array

    x = np.asanyarray(x)

    if x.ndim != 1:
        raise ValueError('Only 1D arrays supported')

    n = x.shape[0]

    # Handle empty array

    if n == 0:
        return np.array([]), np.array([]), np.array([])

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

        run_ends = run_starts + run_lengths

        return run_starts, run_ends


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def get_scores(pred_train,
                    pred_test,
                    true,
                    q=1e-3,
                    level=0.8):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """

    # true_reduced =\
    #     np.any(np.greater_equal(true, 1), axis=1).astype(np.uint8)

    true = true.astype(np.uint8)

    # SPOT object
    spot = SPOT(q)

    # data import
    spot.fit(pred_train, pred_test)

    # initialization step
    spot.initialize(level=level,
                    min_extrema=False,
                    verbose=True)

    # run
    ret = spot.run()

    pred = np.zeros_like(pred_test, dtype=np.uint8)

    pred[ret['alarms']] = 1

    pred = adjust_predicts(pred, true, 0.1)

    precision,\
        recall,\
        f1, _ = precision_recall_fscore_support(true,
                                                    pred,
                                                    average='binary')

    mcc = matthews_corrcoef(true, pred)

    auroc = roc_auc_score(true, pred)

    print(f'AUROC: {auroc:.3f}\t'
            f'F1: {f1:.3f}\t'
            f'MCC: {mcc:.3f}\t'
            f'Precision: {precision:.3f}\t'
            f'Recall: {recall:.3f}')

    return auroc,\
            f1, mcc,\
            precision,\
            recall


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Informer-MSE/Informer-SMSE Anomaly Detection')

    parser.add_argument('--data', type=str, required=True, choices=['HLT_2018', 'HLT_2022', 'HLT_2023', 'SMD'], default='HLT', help='data')
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

    if args.apply_augmentations:

        augmentation_string = ''
        
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
        

    if args.augmented_data_ratio == 0:
        augmentation_string = 'no_augment'

    augment_label = '_no_augment_' if augmentation_string == 'no_augment' else '_'

    setting = f'{args.data.lower()}_{args.loss.lower()}_{augmentation_string}_seed_{int(args.seed)}'

    nas_config = {'loss': tune.choice(['MSE', 'SMSE']),
                    'factor': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    'label_len': tune.choice([4, 8, 16, 32, 64, 128]),
                    'n_heads': tune.choice([3, 4, 5, 6, 7, 8, 9, 10]),
                    'e_layers': tune.choice([1, 2, 3, 4]),
                    'd_layers': tune.choice([1, 2, 3, 4]),
                    'd_model': tune.sample_from(lambda _: 32*np.random.randint(1, 64)),
                    'd_ff': tune.sample_from(lambda _: 64*np.random.randint(1, 64)),
                    'padding': tune.choice([0, 1]),
                    'no_distil': tune.choice([False, True]),
                    'dropout': tune.choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.5]),
                    'attn': tune.choice(['prob', 'full']),
                    'no_mix': tune.choice([False, True]),
                    'learning_rate': tune.loguniform(1e-5, 1e-3),
                    'batch_size': tune.choice([32, 64, 96, 128]),}

    args.s_layers ='3,1'
    args.epochs = 6
    args.freq = 's'
    args.features = 'M'
    args.target = 'OT'
    args.pred_len = 1
    args.embed = 'timeF'
    args.des = 'nas'
    args.lradj = 'type1'
    args.inverse = False

    augmentation_string += f'_rel_size_{args.augmented_dataset_size_relative}'\
                                        f'_ratio_{args.augmented_data_ratio:.2f}'

    setting = 'parameter_variation_study'

    def run_experiment(nas_config, args):

        args.loss = nas_config['loss']
        args.seq_len = nas_config['label_len']*2
        args.label_len = nas_config['label_len']
        args.n_heads = nas_config['n_heads']
        args.e_layers = nas_config['e_layers']
        args.d_layers = nas_config['d_layers']
        args.d_model = nas_config['d_model']
        args.d_ff = nas_config['d_ff']
        args.factor = nas_config['factor']
        args.padding = nas_config['padding']
        args.no_distil = nas_config['no_distil']
        args.dropout = nas_config['dropout']
        args.attn = nas_config['attn']
        args.no_mix = nas_config['no_mix']
        args.learning_rate = nas_config['learning_rate']
        args.batch_size = nas_config['batch_size']

        args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        # Set experiments
        exp = Exp(args)
        
        exp.train(setting)
        exp.test(setting)

        # Load and preprocess train set results

        preds_all_train = np.load('./results/' + setting + '/preds_all_train.npy')
        preds_all_train = preds_all_train.squeeze()

        trues_all_train = np.load('./results/' + setting + '/true_values_all_train.npy')
        trues_all_train = trues_all_train.squeeze()

        l2_dist_scaler = MinMaxScaler()

        l2_distances_all_train = np.mean((preds_all_train[:, :] - trues_all_train[:, :])**2, 1)

        l2_dist_scaler.fit(l2_distances_all_train.reshape(-1, 1))

        l2_distances_all_normalized_train =\
                l2_dist_scaler.transform(l2_distances_all_train.reshape(-1, 1)).flatten()

        # Load and preprocess test set results

        preds_all_test = np.load('./results/' + setting + '/preds_all_test.npy')
        preds_all_test = preds_all_test.squeeze()

        trues_all_test = np.load('./results/' + setting + '/true_values_all_test.npy')
        trues_all_test = trues_all_test.squeeze()
        l2_distances_all_test = np.mean((preds_all_test[:, :] - trues_all_test[:, :])**2, 1)

        l2_distances_all_normalized_test =\
                l2_dist_scaler.transform(l2_distances_all_test.reshape(-1, 1)).flatten()
        
        label_begin = args.seq_len
        label_end = len(preds_all_test) + args.seq_len

        labels_all_test =\
            np.load('./results/' + setting + '/labels_all_test.npy')[label_begin:label_end]

        spot_train_size = int(len(l2_distances_all_test)*0.1)

        auroc_best = 0
        f1_best = 0
        mcc_best = -1
        precision_best = 0
        recall_best = 0

        for q in np.linspace(0.0005, 0.05, 128):

            auroc,\
                f1, mcc,\
                precision,\
                recall =\
                    get_scores(l2_distances_all_normalized_train[:spot_train_size],
                                                        l2_distances_all_normalized_test,
                                                        labels_all_test, q, 0.8)
            
            if mcc > mcc_best:
                auroc_best = auroc
                f1_best = f1
                mcc_best = mcc
                precision_best = precision
                recall_best = recall

        tune.report(AUROC=auroc_best,
                            F1=f1_best,
                            MCC=mcc_best,
                            Precision=precision_best,
                            Recall=recall_best)

    scheduler = AsyncHyperBandScheduler()

    reporter = CLIReporter(
                metric_columns=['AUROC', 'F1', 'MCC', 'Prec' 'Rec'])

    tune_config = tune.TuneConfig(metric='MCC',
                                    mode='max',
                                    scheduler=scheduler,
                                    num_samples=2048,
                                    max_concurrent_trials=1)

    tuner = tune.Tuner(tune.with_resources(
                            tune.with_parameters(
                                partial(run_experiment, args=args)),
                            resources={'cpu': 16, 'gpu': 1}),
                            tune_config=tune_config,
                            param_space=nas_config)
    
    results = tuner.fit()

    best_trial = results.get_best_trial('MCC', 'max')
    print(f'Best trial config: {best_trial.config}')

    rmtree(os.path.join('./results/', setting))
