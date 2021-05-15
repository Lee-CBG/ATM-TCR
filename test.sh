#!/bin/bash


# Number of parallel processes
N=2
i=0
# GPU to run on 
gpu=0
epoch=300

SPLIT=peptide
INFILE=combined_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt
N_FOLD=5

head=5
linear_size=( 1024 )
tcr_size=( 20 )
pep_size=( 15 )

for linear in "${linear_size[@]}"
do
for tcr_len in "${tcr_size[@]}"
do
for pep_len in "${pep_size[@]}"
do
for idx_test_fold in {0..4}
do
#    idx_validation_fold=$(( RANDOM % 5 ))
#    while [ ${idx_test_fold} == ${idx_validation_fold} ]
#    do 
#        idx_validation_fold=$(( RANDOM % 5 ))
#    done
#        # Wait for last N jobs to finish
#        if !(($i % N))
#            then wait
#        fi
#
#        MODELNAME=${INFILE}_split${SPLIT}_foldtest${idx_test_fold}_foldval${idx_validation_fold}_linear${linear}_peplen${pep_len}_tcrlen${tcr_len}_epoch${epoch}.ckpt	
#        # Check if index file exists
#        if [ ! -f "$INDEX_FILE" ]; 
#        then
#            # Index file doesn't exist run single instance alone
#            echo "$INDEX_FILE does not exist, running single instance first"
#            CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold $idx_validation_fold --model_name $MODELNAME --lin_size $linear --heads $head --early_stop True --min_epoch 100 --epoch $epoch --max_len_tcr $tcr_len --max_len_pep $pep_len
#        else
            # Inner cross validation
#            CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold $idx_validation_fold --model_name $MODELNAME --lin_size $linear --heads $head --early_stop True --min_epoch 100 --epoch $epoch --max_len_tcr $tcr_len --max_len_pep $pep_len &
#	        echo "Starting training $i for: $MODELNAME"
#            (i=i+1)
#        fi
    

#Wait for last N jobs to finish
if !(($i % N))
    then wait
fi
# Outer cross validation
MODELNAME=${INFILE}_split${SPLIT}_foldtest${idx_test_fold}_outer_linear${linear}_peplen${pep_len}_tcrlen${tcr_len}_epoch${epoch}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --split_type $SPLIT --idx_test_fold $idx_test_fold --idx_val_fold -1 --model_name $MODELNAME --save_model True --lin_size $linear --heads $head --early_stop True --min_epoch 100 --epoch $epoch --max_len_tcr $tcr_len --max_len_pep $pep_len &
echo "Starting training $i for: $MODELNAME"
((i=i+1))

done #test_fold
done #pep_len
done #tcr_len
done #linear
