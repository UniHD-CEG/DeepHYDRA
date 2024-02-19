#!/usr/bin/env bash

# Predetermined randomized seeds to ensure replicability
declare -a seeds=(7 129 28 192 85 148 142 30 78 33)

for seed in ${seeds[@]}
do
    python3 main.py --seed ${seed} --model DAGMM --dataset "machine-1-1" --retrain
done