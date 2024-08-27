#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
declare -a seeds=(7 129 28 192 85 148 142 30 78 33)

for seed in ${seeds[@]}
do
    python3 main.py --model TranAD --seed=${seed} --dataset "ECLIPSE_MEDIAN" --retrain --apply_augmentations --augmentations 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0'
    python3 main.py --model OmniAnomaly --seed=${seed} --dataset "ECLIPSE_MEDIAN" --retrain --apply_augmentations --augmentations 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0'
    python3 main.py --model DAGMM --seed=${seed} --dataset "ECLIPSE_MEDIAN" --retrain --apply_augmentations --augmentations 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0'
    python3 main.py --model USAD --seed=${seed} --dataset "ECLIPSE_MEDIAN" --retrain --apply_augmentations --augmentations 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0' 'Identity:0'
done