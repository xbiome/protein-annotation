# PO2Vec: partial order relation-based gene ontology embedding improves protein function prediction

PO2Vec is a novel method for GO term representation learning. Compared with existing methods based on GO DAG structure, PO2Vec captures the topological information of GO more comprehensively under in-path partial order constraint and out-path partial order constraint. The effectiveness of PO2Vec was demonstrated in experimental
analyses from five aspects.

A novel protein function annotation prediction model, named PO2GO, which is jointly constructed by PO2Vec and the
protein language pre-trained model ESM-1b. The superior performance of PO2GO is demonstrated with comparative benchmarks.

This repository contains script which were used to build and train the PO2Vec and PO2GO model together with the scripts for evaluating the model's performance.

## Dependencies
* The code was developed and tested using python 3.9.
* To install python dependencies run:
  `pip install -r requirements.txt`



## Data


## Installation
The sources for Deepfold can be downloaded from the `Github repo`.

You can either clone the public repository:

```bash
# clone project
git clone https://github.com/xbiome/protein-annotation
# First, install dependencies
pip install -r requirements.txt
```

Once you have a copy of the source, you can install it with:

```bash
python setup.py install
```

## Running
* Download all the data files and place them into data folder
* `deepgoplus --data-root <path_to_data_folder> --in-file <input_fasta_filename>`


## Scripts
The scripts require GeneOntology in OBO Format.
* uni2pandas.py - This script is used to convert data from UniProt
database format to pandas dataframe.
* deepgoplus_data.py - This script is used to generate training and
  testing datasets.
* deepgoplus.py - This script is used to train the model
* evaluate_*.py - The scripts are used to compute Fmax, Smin and AUPR

The online version of DeepGOPlus is available at http://deepgoplus.bio2vec.net/

## Citation

If you use DeepGOPlus for your research, or incorporate our learning algorithms in your work, please cite:



## New version specifications
Current dependencies can be found in the requirements.txt file.
The used Python version is 3.9.12.


