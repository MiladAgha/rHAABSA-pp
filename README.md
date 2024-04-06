# rHAABSA++
The code for a robuust HAABSA++ model.

## Installation

### Data

The `data/ARTSData` directory contains all the preprocessed data files. 
The files appended with ARTS are the augmented data files. 
The files appended with ont are the ids of the data the ontology reasoner could not classify.

### Setup environment

Create a conda environment with Python version 3.8, the required packages and their versions are listed in `requirements.txt`, note that you may need to install some packages using `conda install` instead of `pip install` depending on your platform.

## Usage

- `main_ont.py`: use this code for running the ontology reasoner
- `main_nn.py`: use this code for running the LCR-rot-hop++
- `BERTget.py` & `BERTget.py`: use this code to build the BERT word embeddings
- `analyseData.py` : use this code to analyse the data and the results

## Acknowledgements

This respository uses code from the following repositories:

- https://github.com/mtrusca/HAABSA_PLUS_PLUS
- https://github.com/google-research/bert
- https://github.com/zhijing-jin/ARTS_TestSet