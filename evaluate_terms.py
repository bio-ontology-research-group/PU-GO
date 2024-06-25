#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os

from collections import Counter
import logging
import copy

from sklearn.metrics import roc_curve, auc, matthews_corrcoef

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='pu',
    help='Prediction model')
def main(data_root, ont, model):
    train_data_file = f"{data_root}/{ont}/train_data.pkl"
    valid_data_file = f"{data_root}/{ont}/valid_data.pkl"
    
    terms_file = f'{data_root}/{ont}/terms.pkl'
    
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    
    base_annots = Counter()
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    for i, row in enumerate(train_df.itertuples()):
        base_annots.update(row.prop_annotations)
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    for i, row in enumerate(valid_df.itertuples()):
        base_annots.update(row.prop_annotations)

    out_data = []
    for run in range(1, 11):
        logger.info(f"Run {run}")
        test_data_file = f'{data_root}/{ont}/predictions_{model}_{run}_time.pkl'
        test_df = pd.read_pickle(test_data_file)
        preds = np.empty((len(test_df), len(terms)), dtype=np.float32)
        labels = np.zeros((len(test_df), len(terms)), dtype=np.float32)
        
        annots = copy.deepcopy(base_annots)
        for i, row in enumerate(test_df.itertuples()):
            preds[i, :] = row.preds
            annots.update(row.prop_annotations)
            for go_id in row.prop_annotations:
                if go_id in terms_dict:
                    labels[i, terms_dict[go_id]] = 1

        total_n = 0
        total_sum = 0
        aucs = []
        anns = []
        used_gos = []
        for go_id, i in terms_dict.items():
            pos_n = np.sum(labels[:, i])
            assert pos_n < len(test_df), f"len test_df: {len(test_df)}"
            if pos_n == 0:
                continue
            total_n += 1
            roc_auc, fpr, tpr = compute_roc(labels[:, i], preds[:, i])
            total_sum += roc_auc
            aucs.append(roc_auc)
            anns.append(annots[go_id])
            used_gos.append(go_id)
        out_data.append((f"aucs_{run}", aucs))
        
        logger.info(f'Average AUC for {ont} in model {run} {total_sum / total_n:.3f}')
    out_data.append(('annots', anns))
    out_data.append(('gos', used_gos))
    
    df = pd.DataFrame(dict(out_data))
    df.to_pickle(f'{data_root}/{ont}/{model}_auc_annots.pkl')
                
    
        
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr

if __name__ == '__main__':
    main()
