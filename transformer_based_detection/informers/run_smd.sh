#!/usr/bin/env bash

# MSE model
python3 informers.py --seed=28 --model="informer" --data="machine-1-1" --features="M" --freq="s" --checkpoints="./checkpoints" --seq_len=64 --label_len=32 --pred_len=1 --enc_in=38 --dec_in=38 --c_out=38 --d_model=512 --n_heads=8 --e_layers=3 --d_layers=2 --d_ff=2048 --factor=5 --padding=0 --dropout=0.05 --loss="MSE" --num_workers=0 --train_epochs=4 --batch_size=64

# SMSE model
python3 informers.py --seed=7 --model="informer" --data="machine-1-1" --features="M" --freq="s" --checkpoints="./checkpoints" --seq_len=64 --label_len=32 --pred_len=1 --enc_in=38 --dec_in=38 --c_out=38 --d_model=512 --n_heads=8 --e_layers=3 --d_layers=2 --d_ff=2048 --factor=5 --padding=0 --dropout=0.05 --loss="SMSE" --num_workers=0 --train_epochs=4 --batch_size=64