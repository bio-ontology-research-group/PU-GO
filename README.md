# PU-GO: Predicting Functions with Positive-Unlabeled Learning

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

* The code was developed and tested using Python 3.8
* To install the necessary dependencies run: `conda env create -f environment.yml`
* Install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)

## Data Availability (TODO)

* https://deepgo.cbrc.kaust.edu.sa/data/pugo - Here you can find the data used to train and evaluate our method.
  * `data.tar.gz` - UniProtKB-SwissProt dataset (release 2023_03)
  * `time-data.tar.gz` Testing dataset from UniProtKB-SwissProt version 2023_05.
  * `models.tar.gz` Trained models. 10 models per each subontology: MFO, CCO and BPO.

## Scripts

* `pu.py` script to train PU-GO on the similarity-based split. We show the commands with the selected hyperparameters for each subontology.
  * MFO: `python pu.py -dr data/ -ont mf  --run 0 --batch_size 35 --loss_type pu_ranking_multi --margin_factor 0.01152 --max_lr 0.004573 --min_lr_factor 0.0008792 --prior 0.000143`
  * CCO:  `python pu.py -dr data/ -ont cc  --run 0 --batch_size 30 --loss_type pu_ranking_multi --margin_factor 0.09812 --max_lr 0.0008681 --min_lr_factor 0.08364 --prior 0.0001106`
  * BPO: `python pu.py -dr data/ -ont bp  --run 0 --batch_size 39 --loss_type pu_ranking_multi --margin_factor 0.02457 --max_lr 0.0004305 --min_lr_factor 0.09056 --prior 0.0007845`


* `pu_time.py` script to train PU-GO on the time-based split. For each command before, just add `-ld` to load the model.

* `predict.py` script to predict functions given a FASTA file. 
  * Usage: `python predict.py -if your_fasta_file.fa -d cuda`

Note: We used the ESM2-15B model using a NVIDIA A100 GPU.

## Citation (TODO)
