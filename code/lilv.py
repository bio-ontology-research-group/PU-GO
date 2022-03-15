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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DeepGODataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X, self.Y = X, Y

    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y


class DeepGOPU(torch.nn.Module):
    def __init__(self, emb_dim, terms_dict, iprs_dict, do, prior, pu, lmbda):
        super().__init__()
        self.emb_dim = emb_dim
        self.do = torch.nn.Dropout(do)
        self.prior = prior
        self.pu = pu
        self.fc_1 = torch.nn.Linear(100, emb_dim)
        self.fc_2 = torch.nn.Linear(emb_dim, len(terms_dict))
        self.lmbda = lmbda

        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_2.weight.data)

    def pur_loss(self, pred, label):
        p_above = - (torch.nn.functional.logsigmoid(pred) * label).sum() / label.sum()
        p_below = - (torch.nn.functional.logsigmoid(-pred) * label).sum() / label.sum()
        u = - torch.nn.functional.logsigmoid((pred * label).sum() / label.sum() - (pred * (1 - label)).sum() / (1 - label).sum())
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above   

    def pu_loss(self, pred, label, lmbda):
        pos_label = (label == 1).float()
        unl_label = (label == 0).float() + (label == -1).float()

        p_above = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        p_below = (torch.log(1 - torch.sigmoid(pred) + 1e-10) * pos_label).sum() / pos_label.sum()
        u = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * unl_label).sum() / unl_label.sum()
        if u > self.prior * p_below:
            return self.prior * p_above - self.prior * p_below + u
        else:
            return self.prior * p_above


        # pos_label = (label == 1).float()
        # unl_label = (label == 0).float()
        # # unl_label = unl_label * (torch.randn_like(unl_label).uniform_() < lmbda)
        # neg_label = (label == -1).float()

        # p_above = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        # p_below = (torch.log(1 - torch.sigmoid(pred) + 1e-10) * pos_label).sum() / pos_label.sum()
        # u_0 = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * unl_label).sum() / unl_label.sum()
        # if neg_label.sum() > 0:
        #     u_1 = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * neg_label).sum() / neg_label.sum()
        #     u = (u_0 + u_1) / 2
        # else:
        #     u = u_0
        # if u > self.prior * p_below:
        #     return self.prior * p_above - self.prior * p_below + u
        # else:
        #     return self.prior * p_above

        # # PU Learning
        # p_above = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        # p_below = (torch.log(1 - torch.sigmoid(pred) + 1e-10) * pos_label).sum() / pos_label.sum()
        # u = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * unl_label).sum() / unl_label.sum()
        # if u > self.prior * p_below:
        #     pu_loss = self.prior * p_above - self.prior * p_below + u
        # else:
        #     pu_loss = self.prior * p_above

        # # PN Learning
        # if neg_label.sum() > 0:
        #     # pn_loss = - (torch.nn.functional.logsigmoid(pred) * pos_label).sum() / pos_label.sum() - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * neg_label).sum() / neg_label.sum()
        #     pn_loss = - torch.nn.functional.logsigmoid((pred * pos_label).sum() / pos_label.sum() - (pred * neg_label).sum() / neg_label.sum())
        # else:
        #     pn_loss = 0
        # return pu_loss + lmbda * pn_loss
    
    def pn_loss(self, pred, label):
        pos = - (torch.nn.functional.logsigmoid(pred) * label).sum() / label.sum()
        neg = - (torch.log(1 - torch.sigmoid(pred) + 1e-10) * (1 - label)).sum() / (1 - label).sum()
        return pos + neg

    def forward(self, X, Y):
        X_pred = self.fc_2(torch.nn.functional.relu(self.do(self.fc_1(X))))
        if self.pu == 0:
            return self.pn_loss(X_pred, Y)
        elif self.pu == 1:
            return self.pu_loss(X_pred, Y, self.lmbda)
        elif self.pu == 2:
            return self.pur_loss(X_pred, Y)
        
    def predict(self, X):
        return torch.sigmoid(self.fc_2(torch.nn.functional.relu(self.do(self.fc_1(X)))))

def read_neg(root):
    info = []
    with open(root + 'neg.txt') as f:
        for line in f:
            line = line.split('\t')
            if line[0] == 'UniProtKB' and 'NOT' in line[3]:
                info.append([line[1], line[4]])
    info = pd.DataFrame(info, columns=['protein', 'terms'])
    info_grouped = info.groupby(['protein'])['terms'].apply(list).reset_index(name='grouped').values
    info_dict = {}
    for record in info_grouped:
        info_dict[record[0]] = record[1]
    return info_dict

def read_data(root):
    terms = pd.read_pickle(root + 'terms.pkl')['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    iprs = pd.read_pickle(root + 'interpros.pkl')['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    go = Ontology(cfg.root + 'go.obo', with_rels=True)

    train_data = pd.read_pickle(root + 'train_data.pkl')[[cfg.dataset + '_dl2vec', 'prop_annotations', 'accessions']]
    valid_data = pd.read_pickle(root + 'valid_data.pkl')[[cfg.dataset + '_dl2vec', 'prop_annotations']]
    test_data = pd.read_pickle(root + 'test_data.pkl')[[cfg.dataset + '_dl2vec', 'prop_annotations']]

    neg = read_neg(cfg.root)

    X_train, Y_train = name2idx(train_data.values, terms_dict, iprs_dict, go, neg=neg)
    X_valid, Y_valid = name2idx(valid_data.values, terms_dict, iprs_dict, go)
    X_test, Y_test = name2idx(test_data.values, terms_dict, iprs_dict, go)

    return terms_dict, iprs_dict, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, test_data, go

def name2idx(data, terms_dict, iprs_dict, go, neg=[]):
    X = []
    Y = []
    for i in range(len(data)):
        # xs_mapped = torch.zeros(1, len(iprs_dict))
        # xs = data[i][0]
        # for x in xs:
        #     xs_mapped[0][iprs_dict[x]] = 1
        xs_mapped = torch.tensor(data[i][0]).unsqueeze(dim=0)
        ys_mapped = torch.zeros(1, len(terms_dict))
        ys = data[i][1]
        for y in ys:
            try:
                ys_mapped[0][terms_dict[y]] = 1
            except:
                pass
        if len(neg):
            accessions = data[i][2].strip(';').split('; ')
            for accession in accessions:
                try:
                    negs = neg[accession]
                    for n in negs:
                        try:
                            ys_mapped[0][terms_dict[n]] = -1
                        except:
                            pass
                except:
                    pass
        X.append(xs_mapped)
        Y.append(ys_mapped)
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)

def validate(model, loader, device, verbose):
    if verbose == 1:
        loader = tqdm.tqdm(loader)
    preds = []
    labels = []
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            pred = model.predict(X)
            preds.append(pred)
            labels.append(Y)
    aupr = round(average_precision_score(torch.cat(labels, dim=0).cpu().flatten(), torch.cat(preds, dim=0).cpu().flatten()), 4)
    print(f'#Valid# AUPR: {aupr}')
    return aupr

def test(model, loader, device, verbose, test_data, terms_dict, go):
    if verbose == 1:
        loader = tqdm.tqdm(loader)
    preds = []
    labels = []
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            pred = model.predict(X)
            preds.append(pred)
            labels.append(Y)
    preds = torch.cat(preds, dim=0)

    results = []
    for pred in preds:
        results.append(pred.cpu().numpy().tolist())
    
    print('Propogating results.')
    for scores in results:
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
    
    auc = round(roc_auc_score(torch.cat(labels, dim=0).cpu().flatten(), torch.tensor(results).flatten()), 4)
    aupr = round(average_precision_score(torch.cat(labels, dim=0).cpu().flatten(), torch.tensor(results).flatten()), 4)
    print(f'#Test# AUC: {auc}, AUPR: {aupr}')

    test_data['preds'] = results
    return test_data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../new_data/', type=str)
    parser.add_argument('--dataset', default='mf', type=str)
    # Tunable
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--do', default=0.2, type=float)
    parser.add_argument('--prior', default=0.0001, type=float)
    parser.add_argument('--emb_dim', default=512, type=int)
    parser.add_argument('--pu', default=1, type=int)
    parser.add_argument('--lmbda', default=1, type=float)
    # Untunable
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--valid_interval', default=20, type=int)
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
    save_root = f'../tmp/dataset_{cfg.dataset}_pu_{cfg.pu}_lmbda_{cfg.lmbda}_bs_{cfg.bs}_lr_{cfg.lr}_wd_{cfg.wd}_do_{cfg.do}_prior_{cfg.prior}_emb_dim_{cfg.emb_dim}/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    terms_dict, iprs_dict, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, test_data, go = read_data(root)
    print(f'N Concepts:{len(terms_dict)}\nN Features:{len(iprs_dict)}')
    train_dataset = DeepGODataset(X_train, Y_train)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=cfg.bs, 
                                                   num_workers=cfg.num_workers, 
                                                   shuffle=True, 
                                                   drop_last=True)
    valid_dataset = DeepGODataset(X_valid, Y_valid)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                                   batch_size=cfg.bs, 
                                                   num_workers=cfg.num_workers, 
                                                   shuffle=False, 
                                                   drop_last=False)     
    test_dataset = DeepGODataset(X_test, Y_test)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=cfg.bs, 
                                                  num_workers=cfg.num_workers, 
                                                  shuffle=False, 
                                                  drop_last=False)                                 
    model = DeepGOPU(cfg.emb_dim, terms_dict, iprs_dict, cfg.do, cfg.prior, cfg.pu, cfg.lmbda)
    model = model.to(device)
    tolerance = cfg.tolerance
    max_aupr = 0
    # min_loss = 100000
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
        avg_loss = round(sum(avg_loss)/len(avg_loss), 6)
        print(f'Loss: {avg_loss}')
        if (epoch + 1) % cfg.valid_interval == 0:
            model.eval()
            aupr = validate(model, valid_dataloader, device, cfg.verbose)
            if aupr > max_aupr:
                max_aupr = aupr
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
            # if avg_loss < min_loss:
            #     min_loss = avg_loss
            #     tolerance = cfg.tolerance
            # else:
            #     tolerance -= 1
            torch.save(model.state_dict(), save_root + (str(epoch + 1)))
        if tolerance == 0:
            print(f'Best performance at epoch {epoch - cfg.tolerance * cfg.valid_interval + 1}')
            model.eval()
            model.load_state_dict(torch.load(save_root + str(epoch - cfg.tolerance * cfg.valid_interval + 1)))
            # model.load_state_dict(torch.load(save_root + '420'))
            test_df = test(model, test_dataloader, device, cfg.verbose, test_data, terms_dict, go)
            evaluate(cfg.root[:-1], cfg.dataset, test_df)
            break

