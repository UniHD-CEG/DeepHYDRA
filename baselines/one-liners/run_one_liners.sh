#!/usr/bin/env bash

python3 one_liner_test_hlt_datasets.py
python3 one_liner_test_hlt_dataset_reduced.py
python3 one_liner_test_smd.py
python3 one_liner_test_eclipse_reduced.py --variant 'median'
python3 one_liner_test_eclipse_reduced.py --variant 'mean'
