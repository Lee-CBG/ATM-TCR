<h1 align="center">
    ATM-TCR
</h1>

<br />

## Overview
We demonstrate how the multi-head self-attention structure can be utilized to learn structural information from protein sequences to make binding affinity predictions.

Written using Python 3.8.10

## Usage
The package requirements are detailed in ```requirements.txt```

The input to the model should be specified as a CSV with the following format:
```
epitope,TCR,binding affinity
```
Where epitope and TCR are the linear protein sequences and binding affinity is either 0 or 1.

To train the model on our dataset using the default settings and on the first GPU
```
CUDA_VISIBLE_DEVICES=0 python main.py --infile data/combined_dataset.csv
```

To make a prediction using an already existing model
```
python main.py --infile data/combined_dataset.csv --indepfile data/covid19_data.txt --model_name model.ckpt --mode test
```

## Data
combined_dataset.csv - A dataset sourced from three databases [VDJdb](https://vdjdb.cdr3.net/), [McPAS](http://friedmanlab.weizmann.ac.il/McPAS-TCR/), and [IEDB](https://www.iedb.org/). Further information about the dataset can be found in ```data/data_analytics.ipynb```
covid19_data.csv - A SARS-CoV-2 dataset sourced from [IEDB](https://www.iedb.org/) that is independent of the data found in combined_dataset.csv.