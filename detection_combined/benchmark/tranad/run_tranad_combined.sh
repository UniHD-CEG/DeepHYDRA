#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
declare -a seeds=(7 129 28 192 85 148 142 30 78 33)

for seed in ${seeds[@]}
do
    python3 tranad.py --seed ${seed} --checkpoint-dir "../../../transformer_based_detection/tranad/checkpoints/TranAD_HLT_Scale_1.0_1.2_Scale_APP_1.0_1.2_0.01_0.01_0.01_rel_size_1.0_ratio_0.25_seed_${seed}"
done