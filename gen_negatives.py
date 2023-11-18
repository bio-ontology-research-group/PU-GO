import click as ck
import pandas as pd
import numpy as np
import os


@ck.command()
@ck.option('--data-frame', '-df', help='Output of uni2pandas.py script')
def main(data_frame):
    regulations = load_regulations()

    df = pd.read_pickle(data_frame)
    fn, ext = os.path.splitext(data_frame)
    output = fn + '_negs' + ext
    neg_annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = set()
        for go_id in row.prop_annotations:
            if go_id in regulations:
                annots.add(regulations[go_id])
        neg_annotations.append(annots)
        print(len(annots))
    df['neg_annotations'] = neg_annotations
    df.to_pickle(output)
    print(df)
    
def load_regulations():
    regs = {}
    with open('data-sim/regulations.txt') as f:
        for line in f:
            it = line.strip().split('\t')
            reg, cl, pn = it[1].replace('_', ':'), it[0].replace('_', ':'), it[2]
            if reg not in regs:
                regs[reg] = {}
            regs[reg][pn] = cl
    regulations = {}

    for key, vals in regs.items():
        if 'pos' in vals and 'neg' in vals:
            regulations[vals['pos']] = vals['neg']
            regulations[vals['neg']] = vals['pos']
    return regulations

if __name__ == '__main__':
    main()
