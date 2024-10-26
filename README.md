# Predicting protein functions using positive-unlabeled ranking with ontology-based priors


## Abstract

Automated protein function prediction is a crucial and widely studied
problem in bioinformatics. Computationally, protein function is a
multilabel classification problem where only positive samples are
defined and there is a large number of unlabeled annotations.  Most
existing methods rely on the assumption that the unlabeled set of
protein function annotations are negatives, inducing the \emph{false
negative} issue, where potential positive samples are trained as
negatives. We introduce a novel approach named PU-GO, wherein we
address function prediction as a positive-unlabeled classification
problem. We apply empirical risk minimization, i.e., we minimize the
classification risk of a classifier where class priors are obtained
from the Gene Ontology hierarchical structure. We show that our
approach is more robust than other state-of-the-art methods on
similarity-based and time-based benchmark datasets


## Dependencies

* Python 3.10
* [PyTorch 2.1.0](https://pytorch.org/)
* [FAIR-ESM](https://github.com/facebookresearch/esm) (for predicting over FASTA files)
* Other basic dependencies can be installed by running: `conda env create -f environment.yml`
* Install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)

## Data Availability

* Download our data from https://dx.doi.org/10.5281/zenodo.11079885 - Here you can find the data used to train and evaluate our method.
* You will get a file named `pu-go-data.tar.gz`. Place it under the directory containing this repository
* Uncompress the data with `tar -xzvf pu-go-data.tar.gz`

* For each subontology directory [mf, cc, bp] run the following command to extract the 10 trained models:

```
cd data/mf
tar -xzvf models.tar.gz
```

## Scripts

* `pu_go.py` script to train/test PU-GO on the similarity-based split. We show the commands with the selected hyperparameters for each subontology.
  * MFO: `python pu_go.py -dr data/ -ont mf  --run 0 --batch_size 35 --loss_type pu_ranking_multi --margin_factor 0.01152 --max_lr 0.004573 --min_lr_factor 0.0008792 --prior 0.000143 -ld`
  * CCO:  `python pu_go.py -dr data/ -ont cc  --run 0 --batch_size 30 --loss_type pu_ranking_multi --margin_factor 0.09812 --max_lr 0.0008681 --min_lr_factor 0.08364 --prior 0.0001106 -ld`
  * BPO: `python pu_go.py -dr data/ -ont bp  --run 0 --batch_size 39 --loss_type pu_ranking_multi --margin_factor 0.02457 --max_lr 0.0004305 --min_lr_factor 0.09056 --prior 0.0007845 -ld`


* `pu_go_time.py` script to train PU-GO on the time-based split. 

* `predict.py` script to predict functions given a FASTA file. 
  * Usage: `python predict.py -if your_fasta_file.fa -d cuda`

Note: We used the ESM2-15B model using a NVIDIA A100 GPU.

## Citation (TODO)
