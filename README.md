<h1 align="center">
    ATM-TCR
</h1>

## Overview
We demonstrate how a multi-head self-attention based model can be utilized to learn structural information from protein sequences to make binding affinity predictions.

## Requirements
Written using Python 3.8.10

The pip package dependencies are detailed in ```requirements.txt```

To install directly from the requirements list
```
pip install -r requirements.txt
```
It is recommended you utilize a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

## Input Format
The input to the model should be specified as a CSV with the following format:
```
epitope,TCR,binding affinity
```

Where epitope and TCR are the linear protein sequences and binding affinity is either 0 or 1.

Example:
```
GLCTLVAML,CASSEGQVSPGELF,1
GLCTLVAML,CSATGTSGRVETQYF,0
```

## Training
To train the model on our dataset using the default settings and on the first GPU
```
CUDA_VISIBLE_DEVICES=0 python main.py --infile data/combined_dataset.csv
```

To change the device to be utilized for training change the ```CUDA_VISIBLE_DEVICES``` to the device number as indicated by ```nvidia-smi```.

The default model name utilized by the program is  ```original.ckpt```. To change the outputted/read model name utilize the following optional argument:
```
--model_name my_custom_model_name
```

After training has finished the model will appear under the ```models``` folder under ```model_name.ckpt``` and two csv files will appear in the ```result``` folder. These files will be called ```perf_model_name.csv``` and ```pred_model_name.csv``` respectively.

```perf_model_name.csv``` contains the a description of performance metrics throughout training. Each line of the csv is the performance of the training model on the validation set in that particular epoch. The last line of the file contains the final performance statistics. An example along with the parameter names:
```
Loss Accuracy Precision1 Precision0 Recall1 Recall0 F1Macro F1Micro AUC
37814.6235	0.6101	0.6241	0.5988	0.5542	0.666	0.6089	0.6101	0.6749
```

```pred_model_name.csv``` contains the predictions of the model on the validation set of data. Each line is a pair from the validation set along with the label and prediction made by the model. The calculated score from the model is also included.
```
Epitope                 TCR	             Actual Pred Binding Affinity
GLCTL@@@@@@@@@@@VAML	CASCWN@@@@@@@YEQYF	1	1	0.9996516704559326
```
## Testing
To make a prediction using a pre-trained model
```
python main.py --infile data/combined_dataset.csv --indepfile data/covid19_data.txt --model_name my_custom_model_name --mode test
```

The predictions will be saved into the ```result``` folder under the name ```pred_model_name_indep_test_data.csv```. These will be displayed similarly to the validation set predictions made during training.

## Optional Arguments

For more information on optional hyperparameter and training arguments
```
python main.py --help
```

## Data

See the README inside of the data folder for additional information.