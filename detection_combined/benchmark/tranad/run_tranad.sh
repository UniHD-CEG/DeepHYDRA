#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
declare -a seeds=(7 129 28 192 85 148 142 30 78 33)

for seed in ${seeds[@]}
do
    #python3 run.py --seed ${seed} --checkpoint-dir "../../../transformer_based_detection/tranad/checkpoints/TranAD_HLT_Scale_1.0_1.2_Scale_APP_1.0_1.2_0.01_0.01_0.01_rel_size_1.0_ratio_0.25_seed_${seed}"
    python3 run.py --model "DAGMM" --seed ${seed} --checkpoint-dir "../../../transformer_based_detection/tranad/checkpoints/DAGMM_HLT_DCM_2018_Scale_1.0_1.2_Scale_APP_1.0_1.2_0.01_0.01_0.01_rel_size_1.0_ratio_0.25_seed_${seed}"
    python3 run.py --model "OmniAnomaly" --seed ${seed} --checkpoint-dir "../../../transformer_based_detection/tranad/checkpoints/OmniAnomaly_HLT_DCM_2018_Scale_1.0_1.2_Scale_APP_1.0_1.2_0.01_0.01_0.01_rel_size_1.0_ratio_0.25_seed_${seed}"
    python3 run.py --model "USAD" --seed ${seed} --checkpoint-dir "../../../transformer_based_detection/tranad/checkpoints/USAD_HLT_DCM_2018_Scale_1.0_1.2_Scale_APP_1.0_1.2_0.01_0.01_0.01_rel_size_1.0_ratio_0.25_seed_${seed}"
done