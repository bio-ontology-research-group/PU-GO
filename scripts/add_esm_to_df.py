#!/usr/bin/env python
import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from extract_esm import extract_esm
import torch as th

@ck.command()
@ck.option('--data_root', '-dr', default='data/',)
@ck.option('--ont', '-o', default='mf', help='Ontology')    
@ck.option('--device', '-d', default='cpu', help='Device for ESM2 model')
def main(data_root, ont, device):
    in_file = f'{data_root}/{ont}/time_data.pkl'
    out_file = f'{data_root}/{ont}/time_data_esm.pkl'
    df = pd.read_pickle(in_file)
    fasta_file = f'{data_root}/{ont}/time_data.fa'

    df_prots = df['proteins'].values
    
    prots, esm2_data = extract_esm(fasta_file, device=device, out_file=f"{data_root}/{ont}/prot_and_esm.th")

    reordered_esm2_data = []
    for prot in df_prots:
        # find index of prot in prots
        idx = prots.index(prot)
        reordered_esm2_data.append(esm2_data[idx])
    
    esm2_data = list(reordered_esm2_data)
    df['esm2'] = esm2_data
    df.to_pickle(out_file)
    logging.info('Successfully saved %d proteins' % (len(df),) )

if __name__ == '__main__':
    main()
