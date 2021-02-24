#bin/bash

# Number of parallel processes
N=2
i=0
# GPU to run on 
gpu=$1

MODEL=cnn
INFILE=data/combined_dataset.csv
INDEX_FILE=data/combined_dataset/tcr_data_shuffle.txt
N_FOLD=10

cnn_size=( 64 128 )
linear_size=( 32 64 )
drop_rate=( 0.3 0.5 )
blosum_matrix=( "/data/blosum/BLOSUM50" "/data/blosum/BLOSUM45" "None" )


for cnn in "${cnn_size[@]}"
do
for linear in "${linear_size[@]}"
do
for drop in "${drop_rate[@]}"
do
for blosum in "${blosum_matrix[@]}"
do
for idx_test_fold in {0..9}
do
for idx_validation_fold in {0..9}
do
    if [ ${idx_test_fold} != ${idx_validation_fold} ]
    then
        # Wait for last N jobs to finish
        if !(($i % N))
            then wait
        fi
        # Check if index file exists
        if [ ! -f "$INDEX_FILE" ]; 
        then
            # Index file doesn't exist run single instance alone
            echo "$INDEX_FILE does not exist."
        else
            # Inner cross validation
            #MODELNAME=${INFILE}_model${MODEL}_foldtest${idx_test_fold}_foldval${idx_validation_fold}_blosum${blosum}_drop${drop}_hid${cnn}_linear${linear}.ckpt
            # Run
            #python main.py --infile data/${INFILE}.txt --blosum $blosum --model $MODEL --model_name $MODELNAME --n_fold $NFOLD --idx_test_fold $idx_test_fold --idx_val_fold $idx_validation_fold --drop_rate $drop --n_hid $cnn --lin_size $linear
            sleep 2
            # Run training
	        echo "starting training $i for: $cnn $linear $drop $blosum"
            ((i=i+1))
        fi
    fi
done
# Wait for last N jobs to finish
if !(($i % N))
    then wait
fi
# Outer cross validation
#MODELNAME=${INFILE}_model${MODEL}_foldtest${IDXFOLDTEST}_foldval${IDXFOLDVAL}_lenpep${MAXLENPEP}_lentcr${MAXLENTCR}_blosum${BLOSUMNUM}_batch${BATCH}_epoch${EPOCH}_pad${PADDING}_drop${DROPOUT}_hid${HID}_filter${FILTER}.ckpt
# Run
#python main.py --infile data/${INFILE}.txt --indepfile data/${INDEPFILE}.txt --blosum data/BLOSUM$BLOSUMNUM --model $MODEL --model_name $MODELNAME --n_fold $NFOLD --idx_test_fold $IDXFOLDTEST --idx_val_fold $IDXFOLDVAL --max_len_pep $MAXLENPEP --max_len_tcr $MAXLENTCR --batch_size $BATCH --epoch $EPOCH --lr $LR --padding $PADDING --drop_rate $DROPOUT --n_hid $HID --n_filters $FILTER
((i=i+1))
done
done
done
done
done