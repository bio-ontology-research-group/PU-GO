import torch
import pandas as pd
import pdb
import tqdm
import os 
import numpy as np
import argparse
import random
from utils import Ontology
from evaluate import evaluate

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, terms_dict, X, Y, num_ng, nf1, nf1_dict, zero_classes):
        super().__init__()
        self.terms_dict = terms_dict
        self.X, self.Y = X, Y
        self.num_ng = num_ng
        self.nf1 = nf1
        self.nf1_dict = nf1_dict
        self.zero_classes = zero_classes
        self.all_c = {**terms_dict, **zero_classes}
    
    def unlabeled_sampling(self, ys):
        y_neg_pool = torch.ones(len(self.terms_dict))
        y_neg_pool[ys] = 0
        y_neg_pool = y_neg_pool.nonzero()
        y_neg = y_neg_pool[torch.randint(len(y_neg_pool), (self.num_ng,))]
        return y_neg
    
    def get_subsumption(self):
        pos = self.nf1[torch.randint(len(self.nf1), (1, ))][0]
        y_neg_pool = torch.ones(len(self.all_c))
        y_neg_pool[self.nf1_dict[pos[0].item()]] = 0
        y_neg_pool = y_neg_pool.nonzero()
        y_neg = y_neg_pool[torch.randint(len(y_neg_pool), (self.num_ng,))]
        sub = torch.cat([pos.unsqueeze(dim=0), torch.cat([pos[0].expand_as(y_neg), y_neg], dim=1)], dim=0)
        return sub

    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        ys = self.Y[idx]
        y_pos = ys[torch.randint(len(ys), (1,))].unsqueeze(dim=0)
        y_neg = self.unlabeled_sampling(ys)
        y = torch.cat([y_pos, y_neg], dim=0).squeeze(dim=-1)
        sub = self.get_subsumption()
        return x, y, sub

class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, terms_dict, X, Y):
        super().__init__()
        self.terms_dict = terms_dict
        self.X, self.Y = X, Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.arange(len(terms_dict))
        pos = self.Y[idx]
        return x, y, pos

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, terms_dict, X):
        super().__init__()
        self.terms_dict = terms_dict
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.arange(len(terms_dict))
        return x, y

class PUGO(torch.nn.Module):
    def __init__(self, emb_dim, terms_dict, iprs_dict, do, prior, zero_classes, puc):
        super().__init__()
        self.emb_dim = emb_dim
        self.do = torch.nn.Dropout(do)
        self.prior = prior
        self.terms_embedding = torch.nn.Embedding(len(terms_dict) + len(zero_classes), emb_dim)
        self.fc_1 = torch.nn.Linear(len(iprs_dict), emb_dim * 2)
        self.fc_2 = torch.nn.Linear(emb_dim * 2, emb_dim)
        # self.fc_1_sub = torch.nn.Linear(emb_dim * 2, emb_dim)
        # self.fc_2_sub = torch.nn.Linear(emb_dim, 1)
        self.puc = puc

        torch.nn.init.xavier_uniform_(self.terms_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_2.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_1_sub.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_2_sub.weight.data)
    
    def pur_loss(self, pred):
        p_above = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
        p_below = - torch.nn.functional.logsigmoid(-pred[:, 0]).mean()
        u = - torch.nn.functional.logsigmoid(pred[:, 0].unsqueeze(-1) - pred[:, 1:]).mean()
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above

    def puc_loss(self, pred):
        p_above = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
        p_below = torch.log(1 - torch.sigmoid(pred[:, 0]) + 1e-10).mean()
        u = - torch.log(1 - torch.sigmoid(pred[:, 1:]) + 1e-10).mean()
        return self.prior * p_above - self.prior * p_below + u

    def forward(self, X, Y, Sub):
        X_emb = self.fc_2(torch.nn.functional.relu(self.do(self.fc_1(X))))
        Y_emb = self.terms_embedding(Y)
        pred_pf = (X_emb.unsqueeze(dim=1) * Y_emb).sum(dim=-1)
        c1_emb = self.terms_embedding(Sub[:, :, 0])
        c2_emb = self.terms_embedding(Sub[:, :, 1])
        pred_sub = (c1_emb * c2_emb).sum(dim=-1)
        # pred_sub = self.fc_2_sub(torch.nn.functional.relu(self.do(self.fc_1_sub(torch.cat([c1_emb, c2_emb], dim=-1))))).squeeze(dim=-1)
        if self.puc == 0:
            return self.pur_loss(pred_pf), self.pur_loss(pred_sub)
        else:
            return self.puc_loss(pred_pf), self.puc_loss(pred_sub)
    
    def predict(self, X, Y):
        X_emb = self.fc_2(torch.nn.functional.relu(self.do(self.fc_1(X))))
        Y_emb = self.terms_embedding(Y)
        return (X_emb * Y_emb.squeeze(dim=0)).sum(dim=-1)
        

def read_data(root):
    terms = pd.read_pickle(root + 'terms.pkl')['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    iprs = pd.read_pickle(root + 'interpros.pkl')['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    go = Ontology(cfg.root + 'go.obo', with_rels=True)
    nf1, nf1_dict, zero_classes = load_normal_forms(cfg.root + 'go.norm', terms_dict)

    neg = {}
    with open(cfg.root + 'neg.txt') as f:
        for line in f:
            line = line.split('\t')
            if 'NOT' in line[3]:
                try:
                    neg[line[2].upper()].append(line[4])
                except:
                    neg[line[2].upper()] = [line[4]]

    train_data = pd.read_pickle(root + 'train_data.pkl')[['interpros', 'prop_annotations']]
    valid_data = pd.read_pickle(root + 'valid_data.pkl')[['interpros', 'prop_annotations']]
    test_data = pd.read_pickle(root + 'test_data.pkl')[['interpros', 'prop_annotations']]

    X_train, Y_train = name2idx(train_data.values, terms_dict, iprs_dict, go)
    X_valid, Y_valid = name2idx(valid_data.values, terms_dict, iprs_dict, go)
    X_test, Y_test = name2idx(test_data.values, terms_dict, iprs_dict, go, test=True)

    return terms_dict, iprs_dict, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, test_data, go, nf1, nf1_dict, zero_classes

def load_normal_forms(go_file, terms_dict):
    nf1 = []
    zclasses = {}
    
    def get_index(go_id):
        if go_id in terms_dict:
            index = terms_dict[go_id]
        elif go_id in zclasses:
            index = zclasses[go_id]
        else:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
            index = zclasses[go_id]
        return index
                
    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if line.find('SubClassOf') == -1:
                continue
            left, right = line.split(' SubClassOf ')
            # C SubClassOf D
            if len(left) == 10 and len(right) == 10:
                go1, go2 = left, right
                nf1.append([get_index(go1), get_index(go2)])
    nf1_df = pd.DataFrame(nf1, columns=['c1', 'c2'])
    nf1_grouped = nf1_df.groupby(['c1'])['c2'].apply(list).reset_index(name='c2s').values
    nf1_dict = {}
    for record in nf1_grouped:
        nf1_dict[record[0]] = record[1]
    return torch.tensor(nf1), nf1_dict, zclasses

def name2idx(data, terms_dict, iprs_dict, go, test=False):
    X = []
    Y = []
    for i in range(len(data)):
        xs_mapped = torch.zeros(1, len(iprs_dict))
        xs = data[i][0]
        for x in xs:
            xs_mapped[0][iprs_dict[x]] = 1
        ys_mapped = set()
        ys = data[i][1]
        for y in ys:
            for _ in go.get_anchestors(y):
                try:
                    ys_mapped.add(terms_dict[_])
                except:
                    pass
        if test or (len(ys_mapped) and (xs_mapped.sum() != 0)):
            X.append(xs_mapped)
            Y.append(torch.tensor(list(ys_mapped)))
    return torch.cat(X, dim=0), Y

def validate(model, loader, device, verbose):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    if verbose == 1:
        loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for X, Y, pos in loader:
            X = X.to(device)
            Y = Y.to(device)
            logits = model.predict(X, Y)
            filter_pos = set(pos[0].numpy().tolist())
            ranks = torch.argsort(logits, descending=True)
            for each_pos in pos[0]:
                rank = (ranks == each_pos).nonzero().item() + 1
                ranks_better = ranks[:rank - 1]
                for known_pos in filter_pos:
                    if (ranks_better == known_pos).sum() == 1:
                        rank -= 1
                r.append(rank)
                rr.append(1/rank)
                if rank == 1:
                    h1.append(1)
                else:
                    h1.append(0)
                if rank <= 3:
                    h3.append(1)
                else:
                    h3.append(0)
                if rank <= 10:
                    h10.append(1)
                else:
                    h10.append(0)
    r = int(sum(r)/len(r))
    rr = round(sum(rr)/len(rr), 3)
    h1 = round(sum(h1)/len(h1), 3)
    h3 = round(sum(h3)/len(h3), 3)
    h10 = round(sum(h10)/len(h10), 3)
    print(f'#Valid# MRR: {rr}, H1: {h1}, H3: {h3}')
    return r, rr, h1, h3, h10

def test(model, loader, device, verbose, test_data, terms_dict, go, puc):
    if verbose == 1:
        loader = tqdm.tqdm(loader)
    preds = []
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            logits = model.predict(X, Y).unsqueeze(dim=0)
            preds.append(logits)
    preds = torch.cat(preds, dim=0)
    if cfg.puc == 0:
        preds = (preds - preds.min()) / (preds.max() - preds.min())
    else:
        preds = torch.sigmoid(preds)
    results = []
    for pred in preds:
        results.append(pred.cpu().numpy().tolist())
    
    print('Propogating results.')
    for i, scores in enumerate(results):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = scores[j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                scores[terms_dict[go_id]] = score
    test_data['preds'] = results
    return test_data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data/', type=str)
    parser.add_argument('--dataset', default='mf', type=str)
    # Tunable
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--do', default=0.2, type=float)
    parser.add_argument('--prior', default=0.00001, type=float)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--sub', default=1, type=float)
    parser.add_argument('--puc', default=0, type=int)
    # Untunable
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--valid_interval', default=50, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--tolerance', default=3, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    root = cfg.root + '/' + cfg.dataset + '/'
    save_root = f'../tmp/dataset_{cfg.dataset}_puc_{cfg.puc}_sub_{cfg.sub}_bs_{cfg.bs}_lr_{cfg.lr}_wd_{cfg.wd}_do_{cfg.do}_prior_{cfg.prior}_num_ng_{cfg.num_ng}_emb_dim_{cfg.emb_dim}/'
    # save_root = '../tmp/PUR/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    terms_dict, iprs_dict, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, test_data, go, nf1, nf1_dict, zero_classes = read_data(root)
    print(f'N Concepts:{len(terms_dict)}\nN Features:{len(iprs_dict)}')
    train_dataset = TrainDataset(terms_dict, X_train, Y_train, cfg.num_ng, nf1, nf1_dict, zero_classes)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=cfg.bs, 
                                                   num_workers=cfg.num_workers, 
                                                   shuffle=True, 
                                                   drop_last=True)
    valid_dataset = ValidDataset(terms_dict, X_valid, Y_valid)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                                   batch_size=1, 
                                                   num_workers=cfg.num_workers, 
                                                   shuffle=False, 
                                                   drop_last=True)     
    test_dataset = TestDataset(terms_dict, X_test)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=1, 
                                                  num_workers=cfg.num_workers, 
                                                  shuffle=False, 
                                                  drop_last=True)                                 
    model = PUGO(cfg.emb_dim, terms_dict, iprs_dict, cfg.do, cfg.prior, zero_classes, cfg.puc)
    model = model.to(device)
    tolerance = cfg.tolerance
    max_rr = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:')
        model.train()
        avg_loss = []
        if cfg.verbose == 1:
            train_dataloader = tqdm.tqdm(train_dataloader)
        for X, Y, Sub in train_dataloader:
            X = X.to(device)
            Y = Y.to(device)
            Sub = Sub.to(device)
            loss_pf, loss_sub = model(X, Y, Sub)
            loss = loss_pf + cfg.sub * loss_sub
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss)/len(avg_loss), 4)}')
        if (epoch + 1) % cfg.valid_interval == 0:
            model.eval()
            _, mrr, _, _, _ = validate(model, valid_dataloader, device, cfg.verbose)
            if mrr >= max_rr:
                max_rr = mrr
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
            torch.save(model.state_dict(), save_root + (str(epoch + 1)))
        if tolerance == 0:
            print(f'Best performance at epoch {epoch - cfg.tolerance * cfg.valid_interval + 1}')
            model.eval()
            model.load_state_dict(torch.load(save_root + str(epoch - cfg.tolerance * cfg.valid_interval + 1)))
            # model.load_state_dict(torch.load(save_root + '1500'))
            test_df = test(model, test_dataloader, device, cfg.verbose, test_data, terms_dict, go, cfg.puc)
            evaluate(cfg.root[:-1], cfg.dataset, test_df)
            break

