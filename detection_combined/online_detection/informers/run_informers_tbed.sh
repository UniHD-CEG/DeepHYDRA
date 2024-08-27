#!/usr/bin/env bash

# LD_PRELOAD=/cvmfs/sft.cern.ch/lcg/releases/gcc/11.3.0/x86_64-centos7/lib64/libstdc++.so.6 python3 informers.py --model 'Informer-SMSE' --device 'cpu' --checkpoint-dir ../../../transformer_based_detection/informers/checkpoints/hlt_2022_smse_Scale_0.5_1.5_Scale_APP_0.5_1.5_0.1_0.05_0.05_rel_size_1.0_ratio_0.25_seed_129/ --cluster-configuration-version '2018' --seed 129 --spot-based-detection --log-level 'debug' --anomaly-log-dump-interval 5
LD_PRELOAD=/cvmfs/sft.cern.ch/lcg/releases/gcc/11.3.0/x86_64-el9/lib64/libstdc++.so.6 python3 informers.py --model 'Informer-SMSE' --device 'cpu' --checkpoint-dir ../../../transformer_based_detection/informers/checkpoints/hlt_2023_smse_no_augment_seed_129/ --cluster-configuration-version '2023' --seed 129 --spot-based-detection --log-level 'debug' --anomaly-log-dump-interval 5 --gradio-address '188.184.2.97' --gradio-user 'test' --gradio-password 'test'

