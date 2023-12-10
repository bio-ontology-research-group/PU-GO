#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter, deque
from utils import Ontology, FUNC_DICT, NAMESPACES
import logging
import torch

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--go-file', '-gf', default='data/go-basic.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--old-data-file', '-odf', default='data/swissprot_exp_2023_03.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--new-data-file', '-ndf', default='data/swissprot_exp_2023_05.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
def main(data_root, go_file, old_data_file, new_data_file):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')
    
    df = pd.read_pickle(old_data_file)
    new_df = pd.read_pickle(new_data_file)
    print("DATA FILES", len(df), len(new_df))
    
    logging.info('Processing annotations')

    annotations = list()

    for ont in ['cc', 'bp', 'mf']:
        index = []
        for i, row in enumerate(df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if term != FUNC_DICT[ont] and go.get_namespace(term) == NAMESPACES[ont]:
                    ok = True
            if ok:
                index.append(i)
            
        tdf = df.iloc[index]
        train_prots = set(tdf['proteins'])
        index = []
        for i, row in enumerate(new_df.itertuples()):
            ok = False
            for term in row.prop_annotations:
                if term != FUNC_DICT[ont] and go.get_namespace(term) == NAMESPACES[ont]:
                    ok = True
            if ok:
                index.append(i)
        ndf = new_df.iloc[index]
        time_df = ndf[~ndf['proteins'].isin(train_prots)]
        
        time_df.to_pickle(f'data/{ont}/time_data.pkl')
        print(len(time_df))
        

if __name__ == '__main__':
    main()
