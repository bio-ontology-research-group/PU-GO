import click as ck
import pandas as pd

@ck.command()
@ck.option('--data-frame', '-df', help='Swissprot pkl')
def main(data_frame):
    # load interpro2go mappings
    interpro2go = {}
    gos = set()
    
    with open('data/interpro2go.txt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            it = line.strip().split()
            ipr_id = it[0].split(':')[1]
            go_id = it[-1]
            if ipr_id not in interpro2go:
                interpro2go[ipr_id] = set()
            interpro2go[ipr_id].add(go_id)
            gos.add(go_id)

    print(len(gos))
    df = pd.read_pickle(data_frame)
    


if __name__ == '__main__':
    main()
