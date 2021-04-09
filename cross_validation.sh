#!/bin/bash

# Number of parallel processes
N=2
i=0
# GPU to run on 
gpu=$1

SPLIT=tcr
INFILE=combined_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt
N_FOLD=10
MODEL="cnn_attn"
cnn_size=( 64 128 256 )
linear_size=( 32 64 128 )
drop_rate=( 0.3 0.5 )
blosum_matrix=( "62" "50" "none" )

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
#for idx_validation_fold in {0..9}
#do
    # Pick random validation fold
    idx_validation_fold = $((( RANDOM % 10 ) + 1 ))
    while [ ${idx_test_fold} == ${idx_validation_fold} ]
    do 
        idx_validation_fold = $((( RANDOM % 10 ) + 1 ))
    done
    then
        # Wait for last N jobs to finish
        if !(($i % N))
            then wait
        fi
        # Check if index file exists
        if [ ! -f "$INDEX_FILE" ]; 
        then
            # Index file doesn't exist run single instance alone
            echo "$INDEX_FILE does not exist, running single instance first"
            CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold $idx_validation_fold --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear
        else
            # Inner cross validation
            MODELNAME=${INFILE}_split${SPLIT}_foldtest${idx_test_fold}_foldval${idx_validation_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}.ckpt
            CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold $idx_validation_fold --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear &
	        echo "Starting training $i for: $MODELNAME"
            ((i=i+1))
            
        fi
    fi
#done
# Wait for last N jobs to finish
if !(($i % N))
    then wait
fi
# Outer cross validation
MODELNAME=${INFILE}_split${SPLIT}_outerloop_foldtest${idx_test_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --blosum $blosum_path --model_name $MODELNAME --drop_rate $drop --n_hid $cnn --lin_size $linear &
echo "Starting training $i for: $MODELNAME"
((i=i+1))
done
done
done
done
done
