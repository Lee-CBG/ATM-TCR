#!/bin/bash

# Number of parallel processes
gpu=0
SPLIT=tcr
INFILE=combined_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt
N_FOLD=5

blosum_path="none"
drop=0.3
linear=32
tcr_len=15
pep_len=15
heads=5

# Outer cross validation
#modeltype="cnn"
#MODELNAME=${INFILE}_split${SPLIT}_outerloop_foldtest${idx_test_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}_head${i}.ckpt
#CUDA_VISIBLE_DEVICES=$gpu python main.py --model ${modeltype} --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear --early_stop False

modeltype="cnn_attn"
for test_fold in {0..4} do
MODELNAME=${INFILE}_split${SPLIT}_outerloop_foldtest${test_fold}_blosum${blosum_path}_drop${drop}_linear${linear}_head${heads}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --model ${modeltype} --infile data/${INFILE} --split_type $SPLIT --idx_test_fold test_fold --idx_val_fold -1 --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --lin_size $linear --heads $heads --early_stop False --epoch 50 --max_len_tcr $tcr_len --max_len_pep $pep_len
done
