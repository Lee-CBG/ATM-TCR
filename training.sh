#!/bin/bash

# Number of parallel processes
N=1
i=0
# GPU to run on 
gpu=0

SPLIT=random
INFILE=combined_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt
N_FOLD=10
EPOCHS=200
cnn_size=( 512 1024 )
linear_size=( 256 )
drop_rate=( 0.5 )
blosum_matrix=( "none" )

for cnn in "${cnn_size[@]}"
do
for linear in "${linear_size[@]}"
do
for drop in "${drop_rate[@]}"
do
for blosum in "${blosum_matrix[@]}"
do
if [[ "$blosum" != "none" ]]
then
    blosum_path="data/blosum/BLOSUM${blosum}"
else
    blosum_path="none"
fi
for idx_test_fold in {0..4}
do
# Wait for last N jobs to finish
if !(($i % N))
    then wait
fi
# Outer cross validation
MODELNAME=${INFILE}_split${SPLIT}_foldtest${idx_test_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear --epoch $EPOCHS &
echo "Starting training $i for: $MODELNAME"
((i=i+1))
done
done
done
done
done
