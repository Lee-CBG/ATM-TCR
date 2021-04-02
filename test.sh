#!/bin/bash

# Number of parallel processes
gpu=1
SPLIT=tcr
INFILE=combined_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt
N_FOLD=5


idx_test_fold=1
blosum_path="none"
drop=0.3
cnn=128
linear=32

# Outer cross validation
#modeltype="cnn"
#MODELNAME=${INFILE}_split${SPLIT}_outerloop_foldtest${idx_test_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}_head${i}.ckpt
#CUDA_VISIBLE_DEVICES=$gpu python main.py --model ${modeltype} --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear --early_stop False

modeltype="cnn_attn"
for i in 5 1; do
MODELNAME=${INFILE}_split${SPLIT}_outerloop_foldtest${idx_test_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}_head${i}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --model ${modeltype} --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear --heads $i --early_stop False --epoch 50 --max_len_tcr 20 --max_len_pep 20
done
