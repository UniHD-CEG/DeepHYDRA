#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
declare -a seeds=(7 129 28 192 85 148 142 30 78 33)

for seed in ${seeds[@]}
do
    python3 informers.py --seed ${seed} --model 'Informer-MSE' --checkpoint-dir "../../../transformer_based_detection/informers/checkpoints/mse_scale_0.8_1.0_scale_app_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25_${seed}"
    python3 informers.py --seed ${seed} --model 'Informer-SMSE' --checkpoint-dir "../../../transformer_based_detection/informers/checkpoints/smse_scale_0.5_1.5_scale_app_0.5_1.5_0.1_0.05_0.05_rel_size_1.0_ratio_0.25_${seed}"
done