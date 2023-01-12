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


## Scripts
The scripts require GeneOntology in OBO Format.
* preprocess_raw_data.py - This script is used to convert data from UniProt database format to pandas dataframe.
* prepare_data_seperate_swissprot.py - This script is used to seperate data from UniProt-Swissprot into train and test dataset.
* prepare_data_seperate_ontologies.py - This script is used to seperate train(test) data into three groups based on annotation ontologies.
* po2go/po2vec - This folder is used to train PO2Vec model to get terms embeddings.
* train_po2go.py - This script is used to train the model.
- to train a model predict mfo terms run sh: 'python train_po2go.py --namespace mfo -ld 512 -p 768'
- to train a model predict bpo terms run sh: 'python train_po2go.py --namespace bpo -ld 768 -p 1280'
- to train a model predict mfo terms run sh: 'python train_po2go.py --namespace mfo -ld 512 -p 896'
- to train a model predict annotated terms run sh: 'python train_po2go.py --namespace annotated -ld 1024 -p 2048'
* inference_po2go.py - This script is used to inference test data to get prediction file.
* evaluate_*.py - The scripts are used to compute Fmax, Smin and AUPR.
* inference_fasta.py -This script is used to inference fasta format data.

## Citation

If you use PO2Vec for your research, or incorporate our learning algorithms in your work, please cite:



## New version specifications
Current dependencies can be found in the requirements.txt file.
The used Python version is 3.9.12.
