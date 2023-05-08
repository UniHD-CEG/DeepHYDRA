#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
#declare -a seeds=(7 129 28 192 85 148 142 30 78 33)
declare -a seeds=(148 142 30 78 33)
for seed in ${seeds[@]}
do
    python3 informers.py --seed ${seed} --model 'Informer-MSE' --checkpoint-dir "../../../transformer_based_detection/informers/checkpoints/hlt_mse_Scale_0.8_1.0_Scale_APP_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25_seed_${seed}"
    python3 informers.py --seed ${seed} --model 'Informer-SMSE' --checkpoint-dir "../../../transformer_based_detection/informers/checkpoints/hlt_smse_Scale_0.5_1.5_Scale_APP_0.5_1.5_0.1_0.05_0.05_rel_size_1.0_ratio_0.25_seed_${seed}"
done