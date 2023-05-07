#!/usr/bin/env bash

python3 informers.py --model 'Informer-MSE' --checkpoint-dir ../../../transformer_based_detection/informers/checkpoints/mse_scale_0.8_1.0_scale_app_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25
python3 informers.py --model 'Informer-SMSE' --checkpoint-dir ../../../transformer_based_detection/informers/checkpoints/smse_scale_0.5_1.5_scale_app_0.5_1.5_0.1_0.05_0.05_rel_size_1.0_ratio_0.25
