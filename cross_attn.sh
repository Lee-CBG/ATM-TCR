#!/bin/bash

# Number of parallel processes
N=2
i=0
# GPU to run on 
gpu=$2

SPLIT=tcr
INFILE=combined_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt
N_FOLD=10
MODEL="cnn_attn"
linear_size=( 32 64 )
drop_rate=( 0.3 0.5 )
blosum_matrix=( "none" )
heads=( 1 5 )
for linear in "${linear_size[@]}"
do
for drop in "${drop_rate[@]}"
do
for blosum in "${blosum_matrix[@]}"
do
for num_heads in "${heads[@]}"
do
if [[ "$blosum" != "none" ]]
then
    blosum_path="data/blosum/BLOSUM${blosum}"
else
    blosum_path="none"
fi
for idx_test_fold in {0..4}
do
if !(($i % N))
    then wait
fi
# Outer cross validation
MODELNAME=${INFILE}_split${SPLIT}_outerloop_foldtest${idx_test_fold}_blosum${blosum}_drop${drop}_linear${linear}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --model ${MODEL} --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --blosum $blosum_path --heads $num_heads --model_name $MODELNAME --drop_rate $drop --lin_size $linear &
echo "Starting training $i for: $MODELNAME"
((i=i+1))
done
done
done
done
done
