from src.parser import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
        'machine-1-1': [(0.99995, 1.06), (0.99995, 1.06)],
        'machine-2-1': [(0.95, 0.9), (0.95, 0.9)],
        'machine-3-2': [(0.99, 1.), (0.99, 1.)],
        'machine-3-7': [(0.99995, 1.06), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
        'HLT_DCM_2018': [(0.99995, 1.04), (0.99995, 1.06)],
        'HLT_DCM_2022': [(0.99995, 1.04), (0.99995, 1.06)],
        'ECLIPSE_MEDIAN': [(0.99995, 1.04), (0.99995, 1.06)],
		'ECLIPSE_MEAN': [(0.99995, 1.04), (0.99995, 1.06)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001,
        'machine-1-1': 0.0001,
        'machine-2-1': 0.0001,
        'machine-3-2': 0.0001,
        'machine-3-7': 0.0001,
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001, 
        'HLT_DCM_2018': 0.0001,
        'HLT_DCM_2022': 0.0001,
        'ECLIPSE_MEDIAN': 0.0001,
        'ECLIPSE_MEAN': 0.0001,
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
        'machine-1-1': (98, 2000),
        'machine-2-1': (98, 2000),
        'machine-3-2': (98, 2000),
        'machine-3-7': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
        'HLT_DCM_2018': (98, 2000),
        'HLT_DCM_2022': (98, 2000),
        'ECLIPSE_MEDIAN': (98, 2000),
        'ECLIPSE_MEAN': (98, 2000),
	}

percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
