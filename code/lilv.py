import torch
import pandas as pd
import pdb
import tqdm
import os 
import numpy as np
import argparse
import random
from utils import Ontology

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, terms_dict, X, Y, num_ng):
        super().__init__()
        self.terms_dict = terms_dict
        self.X, self.Y = X, Y
        self.num_ng = num_ng
      
    def unlabeled_sampling(self, ys):
        y_neg_pool = torch.ones(len(self.terms_dict))
        y_neg_pool[ys] = 0
        y_neg_pool = y_neg_pool.nonzero()
        y_neg = y_neg_pool[torch.randint(len(y_neg_pool), (self.num_ng,))]
        return y_neg

    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        ys = self.Y[idx]
        y_pos = ys[torch.randint(len(ys), (1,))].unsqueeze(dim=0)
        y_neg = self.unlabeled_sampling(ys)
        y = torch.cat([y_pos, y_neg], dim=0).squeeze(dim=-1)
        return x, y

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
    def __init__(self, emb_dim, terms_dict, iprs_dict, do, prior):
        super().__init__()
        self.emb_dim = emb_dim
        self.terms_embedding = torch.nn.Embedding(len(terms_dict), emb_dim)
        self.fc_1 = torch.nn.Linear(len(iprs_dict), emb_dim * 2)
        self.fc_2 = torch.nn.Linear(emb_dim * 2, emb_dim)
        self.do = torch.nn.Dropout(do)
        self.prior = prior

        torch.nn.init.xavier_uniform_(self.terms_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_2.weight.data)
    
    def pur_loss(self, pred):
        p_above = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
        p_below = - torch.nn.functional.logsigmoid(-pred[:, 0]).mean()
        u = - torch.nn.functional.logsigmoid(pred[:, 0].unsqueeze(-1) - pred[:, 1:]).mean()
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above

    def forward(self, X, Y):
        X_emb = self.fc_2(torch.nn.functional.relu(self.do(self.fc_1(X))))
        Y_emb = self.terms_embedding(Y)
        pred = (X_emb.unsqueeze(dim=1) * Y_emb).sum(dim=-1)
        return self.pur_loss(pred)
    
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

    train_data = pd.read_pickle(root + 'train_data.pkl')[['interpros', 'prop_annotations']].values
    valid_data = pd.read_pickle(root + 'valid_data.pkl')[['interpros', 'prop_annotations']].values
    test_data = pd.read_pickle(root + 'test_data.pkl')[['interpros', 'prop_annotations']].values

    X_train, Y_train = name2idx(train_data, terms_dict, iprs_dict, go)
    X_valid, Y_valid = name2idx(valid_data, terms_dict, iprs_dict, go)
    X_test, Y_test = name2idx(test_data, terms_dict, iprs_dict, go, test=True)

    return terms_dict, iprs_dict, X_train, Y_train, X_valid, Y_valid, X_test, Y_test

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

def validate(model, loader, device):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    if cfg.verbose == 1:
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

def test(model, loader, device):
    pdb.set_trace()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data/', type=str)
    parser.add_argument('--dataset', default='mf', type=str)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--do', default=0.2, type=float)
    parser.add_argument('--prior', default=0.00001, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--valid_interval', default=10, type=int)
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
    save_root = '../tmp/PUGO/'
    
    terms_dict, iprs_dict, X_train, Y_train, X_valid, Y_valid, X_test, Y_test = read_data(root)
    print(f'N Concepts:{len(terms_dict)}\nN Features:{len(iprs_dict)}')
    train_dataset = TrainDataset(terms_dict, X_train, Y_train, cfg.num_ng)
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
    model = PUGO(cfg.emb_dim, terms_dict, iprs_dict, cfg.do, cfg.prior)
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
        for X, Y in train_dataloader:
            X = X.to(device)
            Y = Y.to(device)
            loss = model(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss)/len(avg_loss), 4)}')
        if (epoch + 1) % cfg.valid_interval == 0:
            model.eval()
            _, mrr, _, _, _ = validate(model, valid_dataloader, device)
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
            test(model, test_dataloader, device)
            break




        

