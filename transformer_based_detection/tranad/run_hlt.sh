#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
declare -a seeds=(7 129 28 192 85 148 142 30 78 33)

for seed in ${seeds[@]}
do
    python3 main.py --model TranAD --seed=${seed} --dataset "HLT" --retrain --apply_augmentations --augmentations 'Scale:1.0,1.2' 'Scale_APP:1.0,1.2,0.01,0.01,0.01' --augmented_dataset_size_relative=1.0 --augmented_data_ratio=0.25
done