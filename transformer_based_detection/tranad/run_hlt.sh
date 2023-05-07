#!/usr/bin/env bash

python3 main.py --model TranAD --seed=30 --dataset "HLT" --retrain --apply_augmentations --augmentations 'Scale:1.0,1.2' 'Scale_APP:1.0,1.2,0.01,0.01,0.01' --augmented_dataset_size_relative=1.0 --augmented_data_ratio=0.25
