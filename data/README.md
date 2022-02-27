<h1 align="center">
    Data
</h1>

### blosum
Contains various blosum matrices

### combined_dataset.csv
Our training and testing dataset sourced from three databases [VDJdb](https://vdjdb.cdr3.net/), [McPAS](http://friedmanlab.weizmann.ac.il/McPAS-TCR/), and [IEDB](https://www.iedb.org/). Further information about the dataset creation and contents can be found in ```data_analytics.ipynb```

### covid19_data.csv
A SARS-CoV-2 dataset sourced from [IEDB](https://www.iedb.org/) that is independent of the data found in combined_dataset.csv.

### data_analytics.ipynb
A Jupyter Notebook which contains the summary statistics of the data. It also contains functions for generating negative examples from datasets.

<hr/>

## Original Database Files

### McPAS-TCR.csv
http://friedmanlab.weizmann.ac.il/McPAS-TCR/

Downloaded on December 23rd, 2020

This is the entirety of the McPAS database

### VDJDB.csv
https://vdjdb.cdr3.net/

Downloaded on December 23rd, 2020

VDJDB Search Settings
- CDR3
    - Species: Human, Monkey, Mouse
    - Gene (chain): TRA, TRB
- MHC
    - MHCI, MHCII
- Meta
    - All Assay Types
    - All Sequencing Types
    - Minimal Confidence Score: 0

### iedb_mhc_class_*.csv
https://www.iedb.org/

Downloaded on December 23rd, 2020

The matching files correspond to the MHC classes of the data.

Search Settings for IEDB Data
- Linear Epitope
- Has Receptor Sequence
- Receptor Type TCR αβ
- Positive Assays Only
- T Cell Assays
- MHC Class I AND MHC Class II
- Humans
- Any Disease
- Any Reference Type
