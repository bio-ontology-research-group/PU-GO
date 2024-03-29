#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import time
import math
import gzip
from utils import Ontology, NAMESPACES
from nn import PUModel
from extract_esm import extract_esm
from pathlib import Path
import torch as th
from Bio import SeqIO
import os

@ck.command()
@ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option('--data-root', '-dr', default='data')
@ck.option('--threshold', '-t', default=0.1, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=6, help='Batch size for prediction model')
@ck.option('--device', '-d', default='cpu', help='Device')
def main(in_file, data_root, threshold, batch_size, device):

    # Extract ESM features
    print("Extracting ESM features")
    fn = os.path.splitext(in_file)[0]
    out_file_esm = f'{fn}_esm_embeddings.pkl'
    proteins, data = extract_esm(in_file, out_file=out_file_esm, device=device)
    
    # Load GO and read list of all terms
    go_file = f'{data_root}/go-basic.obo'
    go = Ontology(go_file, with_rels=True)
    ent_models = {
        'mf': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'bp': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'cc': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    
    for ont in ['mf', 'cc', 'bp']:
        print(f'Predicting {ont} classes')
        terms_file = f'{data_root}/{ont}/terms.pkl'
        out_file = f'{fn}_preds_{ont}.tsv.gz'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}

        n_terms = len(terms_dict)

        sum_preds = np.zeros((len(proteins), n_terms), dtype=np.float32)
        model = PUModel(n_terms, None, None, None, None, None, inference=True).to(device)
        for mn in ent_models[ont]:
            model_file = f'{data_root}/{ont}/pu_{mn}.th'
            model.load_state_dict(th.load(model_file, map_location=device))
            model.eval()
            
            with th.no_grad():
                steps = int(math.ceil(len(proteins) / batch_size))
                preds = []
                with ck.progressbar(length=steps, show_pos=True) as bar:
                    for i in range(steps):
                        bar.update(1)
                        start, end = i * batch_size, (i + 1) * batch_size
                        batch_features = data[start:end]
                        batch_features = th.stack(batch_features, dim=0).to(device)
                        logits = model.predict(batch_features)
                        preds.append(logits.detach().cpu().numpy())
                preds = np.concatenate(preds)
            sum_preds += preds
        preds = sum_preds / len(ent_models[ont])
        go_ind = [None] * len(proteins)
        scores = [None] * len(proteins)
        with gzip.open(out_file, 'wt') as f:
            for i in range(len(proteins)):
                above_threshold = np.argwhere(preds[i] >= threshold).flatten()
                for j in above_threshold:
                    name = go.get_term(terms[j])['name']
                    f.write(f'{proteins[i]}\t{terms[j]}\t{preds[i,j]:0.3f}\n')
       
if __name__ == '__main__':
    main()
