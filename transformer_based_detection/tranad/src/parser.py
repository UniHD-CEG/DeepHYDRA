import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--seed', 
                    type=float,
					default=42,
					help="Random seed")
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='HLT_DCM_2018',
                    help="dataset from ['HLT_DCM_2018', 'HLT_DCM_2022', 'HLT_DCM_ECLIPSE', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--apply_augmentations', action='store_true', help='Apply augmentations to the training set')
parser.add_argument('--augmentations', type=str, nargs='+', help='Applied augmentations and the factors used in augmentation')
parser.add_argument('--augmented_dataset_size_relative', type=float, default=1., help='Size of the augmented training set relative to the original training set')
parser.add_argument('--augmented_data_ratio', type=float, default=0., help='Amount of augmented data in the augmented training set')
args = parser.parse_args()
