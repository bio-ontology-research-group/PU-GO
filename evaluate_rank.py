import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
from scipy.stats import rankdata
import math
from utils import FUNC_DICT, Ontology, NAMESPACES, EXP_CODES
from matplotlib import pyplot as plt
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def test(data_root, ont, model, run, combine, alpha, tex_output, wandb_logger):
    
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/predictions_{model}_{run}.pkl'
                    
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go_rels = Ontology(f'{data_root}/go-basic.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)
    
    eval_preds = []
    
    for i, row in enumerate(test_df.itertuples()):
        preds = row.preds
        eval_preds.append(preds)

    labels = np.zeros((len(test_df), len(terms)), dtype=np.float32)
    filtering_labels = np.ones((len(test_df), len(terms)), dtype=np.float32)
    eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))

    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.annotations:
            # locate annotations for protein in train_df
            train_annots = train_df[train_df['proteins'] == row.proteins].annotations.values[0]
            if go_id in terms_dict:
                if go_id in train_annots:
                    filtering_labels[i, terms_dict[go_id]] = 0
                else:
                    labels[i, terms_dict[go_id]] = 1
                    

    # total_n = 0
    # total_sum = 0
    # for go_id, i in terms_dict.items():
        # pos_n = np.sum(labels[:, i])
        # if pos_n > 0 and pos_n < len(test_df):
            # total_n += 1
            # roc_auc  = compute_roc(labels[:, i], eval_preds[:, i])
            # total_sum += roc_auc

    # avg_auc = total_sum / total_n


    mr = 0
    mrr = 0
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    hits_100 = 0
    rank_auc = 0
    ranks = dict()

    n = 0
    for i, row in tqdm(enumerate(test_df.itertuples()), total=len(test_df)):
        for go_id in row.annotations:
            go_id = go_id.split('|')[0]
            if not go_id in terms_dict:
                continue
            go_index = terms_dict[go_id]
            
            n += 1
            preds = row.preds
            filtered_preds = -preds * filtering_labels[i]
            

            ordering = rankdata(filtered_preds, method='average')
            rank = ordering[go_index]
            mr += rank
            mrr += 1 / rank
            if rank == 1:
                hits_1 += 1
            if rank <= 3:
                hits_3 += 1
            if rank <= 10:
                hits_10 += 1
            if rank <= 100:
                hits_100 += 1

            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1
    
    rank_auc = compute_rank_roc(ranks, len(terms_dict))

    mr /= n
    mrr /= n
    hits_1 /= n
    hits_3 /= n
    hits_10 /= n
    hits_100 /= n
    
    
                
    wandb_logger.log({
        "mean_rank": mr,
        "mean_reciprocal_rank": mrr,
        "hits_1": hits_1,
        "hits_3": hits_3,
        "hits_10": hits_10,
        "hits_100": hits_100,
        "rank_auc": rank_auc,
        "total_annots": n,
    })


    
def compute_rank_roc(ranks, n_terms):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_terms)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_terms
    return auc
    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc



if __name__ == '__main__':
    main()
