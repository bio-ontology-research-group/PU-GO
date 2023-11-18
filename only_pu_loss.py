import click as ck
import pandas as pd
from utils import Ontology, seed_everything
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR

from torch_utils import FastTensorDataLoader
import csv
from multiprocessing import Pool, get_context
from functools import partial
import sys
from tqdm import tqdm
import math

from evaluate_new import test
# from evaluate_rank import test


import wandb

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.5, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class DGPROModel(nn.Module):
    def __init__(self, nb_gos, nodes=[2048,]):
        super().__init__()
        self.nb_gos = nb_gos
        input_length = 5120
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)
    
class DeepGOPU(nn.Module):
    def __init__(self, nb_gos, prior, gamma, margin_factor, loss_type, terms_count, device = "cuda"):
        super().__init__()
        self.nb_gos = nb_gos
        self.prior = prior
        self.gamma = gamma
        self.margin = self.prior*margin_factor
        self.dgpro = DGPROModel(nb_gos)
        self.loss_type = loss_type
        self.device = device
        
        max_count = max(terms_count.values())
        print(f"max_count: {max_count}")
        # self.priors = [self.prior*x for x in terms_count.values()]
        self.priors = [self.prior*x/max_count for x in terms_count.values()]
        # self.priors = [min(x/max_count, self.prior) for x in terms_count.values()]
        self.priors = th.tensor(self.priors, dtype=th.float32, requires_grad=False).to(device)
                        
        
    def pu_loss(self, data, labels):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        p_above = - (F.logsigmoid(preds)*pos_label).sum() / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum() / pos_label.sum()
        u_below = - (F.logsigmoid(-preds)*unl_label).sum() / unl_label.sum()

        loss = self.prior * p_above + th.relu(u_below - self.prior*p_below + self.margin)
        return loss

    def pu_ranking_loss(self, data, labels):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        p_above = - (F.logsigmoid(preds)*pos_label).sum() / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum() / pos_label.sum()
        u_below = - (F.logsigmoid(preds * pos_label - preds*unl_label)).sum() / unl_label.sum()

        loss = self.prior * p_above + th.relu(u_below - self.prior*p_below + self.margin)
        return loss

    def pu_loss_multi(self, data, labels):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        
        p_above = - (F.logsigmoid(preds)*pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum(dim=0) / pos_label.sum()
        u_below = - (F.logsigmoid(-preds)*unl_label).sum(dim=0) / unl_label.sum()

        loss = self.priors * p_above + th.relu(u_below - self.priors*p_below + self.margin)
        loss = loss.sum()
        return loss

    def pu_ranking_loss_multi(self, data, labels):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        p_above = - (F.logsigmoid(preds)*pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum(dim=0) / pos_label.sum()
        u_below = - (F.logsigmoid(preds * pos_label - preds*unl_label)).sum(dim=0) / unl_label.sum()

        loss = self.priors * p_above + th.relu(u_below - self.priors*p_below + self.margin)
        loss = loss.sum()
        return loss

    
    def pun_loss(self, data, labels):
        pred = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels == 0).float()
        neg_label = (labels == -1).float()

        p_above = - (F.logsigmoid(pred) * pos_label).sum() / pos_label.sum()
        p_below = - (F.logsigmoid(-pred) * pos_label).sum() / pos_label.sum()
        u_below = - (F.logsigmoid(-pred) * unl_label).sum() / unl_label.sum()

        if neg_label.sum() > 0:
            n_below = - (F.logsigmoid(-pred) * neg_label).sum() / neg_label.sum()
            gamma = self.gamma
        else:
            n_below = 0
            gamma = 0

        
        loss = self.prior * p_above + th.relu(gamma * (1 - self.prior) * n_below +
                                              (1 - gamma) * (u_below - self.prior * p_below + self.margin))
        
        return loss

    def pun_loss_multi(self, data, labels):
        pred = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels == 0).float()
        neg_label = (labels == -1).float()

        #set unl_label values to False with probability p https://arxiv.org/pdf/2308.00279.pdf -> now in pu_base_sample_prior_new.py
        #unl_label = unl_label * (th.rand(unl_label.shape).to(self.device) < p).float()
        #pos_label = pos_label * (th.rand(pos_label.shape).to(self.device) < p).float()
        
        
        p_above = - (F.logsigmoid(pred) * pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-pred) * pos_label).sum(dim=0) / pos_label.sum()
        u_below = - (F.logsigmoid(-pred) * unl_label).sum(dim=0) / unl_label.sum()

        if neg_label.sum() > 0:
            n_below = - (F.logsigmoid(-pred) * neg_label).sum(dim=0) / neg_label.sum()
            gamma = self.gamma # 0.01
        else:
            n_below = 0
            gamma = 0

        margin = 0 #- self.priors / 16
        loss = self.priors * p_above + th.relu(gamma * (1 - self.priors) * n_below +
                                              (1 - gamma) * (u_below - self.priors * p_below + margin))
        
        loss = loss.sum()
        assert loss >= 0, f"loss: {loss}"
        return loss


                                                                 
    
    def forward(self, data, labels):
        # return self.pu_ranking_loss(data, labels)
        if self.loss_type == 'pu':
            return self.pu_loss(data, labels)
        if self.loss_type == 'pu_multi':
            return self.pu_loss_multi(data, labels)
        elif self.loss_type == "pun":
            return self.pun_loss(data, labels)
        elif self.loss_type == "pun_multi":
            return self.pun_loss_multi(data, labels)
        elif self.loss_type == "pu_ranking":
            return self.pu_ranking_loss(data, labels)
        elif self.loss_type == "pu_ranking_multi":
            return self.pu_ranking_loss_multi(data, labels)
        else:
            raise NotImplementedError

    def logits(self, data):
        return self.dgpro(data)
    
    def predict(self, data):
        return th.sigmoid(self.dgpro(data))
 

@ck.command()
@ck.option(
    '--data_root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model_name', '-mn', default='dgpu',
    help='Prediction model')
@ck.option(
    '--batch_size', '-bs', default=256,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--prior', '-p', default=1e-4,
    help='Prior')
@ck.option("--gamma", '-g', default = 0.5)
@ck.option("--alpha", '-a', default = 0.5, help="Weight of the unlabeled loss")
@ck.option('--loss_type', '-loss', default='pu', type=ck.Choice(['pu', 'pun', 'pu_multi', 'pun_multi', 'pu_ranking', 'pu_ranking_multi']))
@ck.option('--max_lr', '-lr', default=1e-4)
@ck.option('--min_lr_factor', '-minlr', default=0.01)
@ck.option('--margin_factor', '-mf', default=0.0)
@ck.option('--load', '-ld', is_flag=True, help='Load Model?')
@ck.option("--alpha_test", "-at", default=0.5)
@ck.option("--combine", "-c", is_flag=True)
@ck.option('--device', '-d', default='cuda', help='Device')
@ck.option('--run', '-r', default='0', help='Run')
def main(data_root, ont, model_name, batch_size, epochs, prior, gamma, alpha, loss_type, max_lr, min_lr_factor,  margin_factor, load, alpha_test, combine, device, run):

                                        
    # seed_everything(42)

    name = f"{ont}_{loss_type}"
    wandb_logger = wandb.init(project="final-dgpu-similarity-based", name= f"{name}_{run}", group=name)
                                    
    go_file = f'{data_root}/go-basic.obo'
    model_name = f"{model_name}_bs{batch_size}_mf{margin_factor}_lr{max_lr}_minlr{min_lr_factor}_p{prior}_r{run}"
    model_file = f'{data_root}/{ont}/{model_name}.th'
    out_file = f'{data_root}/{ont}/predictions_{model_name}_{run}.pkl'


    


    
    go = Ontology(go_file, with_rels=True)
    terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, go)

    n_terms = len(terms_dict)

    
        
    train_features, train_labels, terms_count = train_data
    valid_features, valid_labels, _ = valid_data
    test_features, test_labels, _ = test_data


    net = DeepGOPU(n_terms, prior, gamma, margin_factor, loss_type, terms_count).to(device)
    

    train_loader = FastTensorDataLoader(
        train_features, train_labels, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        valid_features, valid_labels, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        test_features, test_labels, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    print('Labels', np.sum(valid_labels == 1))

    train_steps = int(math.ceil(len(train_labels) / batch_size))
    step_size_up = 2 * train_steps
    
    bce = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(net.parameters(), lr=max_lr)
    #scheduler = MultiStepLR(optimizer, milestones=[1, 3,], gamma=0.1)
    min_lr = max_lr * min_lr_factor
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)
    best_loss = 10000.0
    best_fmax = 0
    tolerance = 5
    curr_tolerance = tolerance
        
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_pu_loss = 0
            train_bce_loss = 0
            
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    pu_loss = net(batch_features, batch_labels)
                    # logits = net.logits(batch_features)

                    batch_labels = (batch_labels == 1).float()
                    #bce_loss = bce(logits, batch_labels)
                    loss = pu_loss
                    # loss = alpha*pu_loss + (1-alpha)*bce_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    train_pu_loss += pu_loss.detach().item()
                    scheduler.step()
            
            train_loss /= train_steps
            train_pu_loss /= train_steps
            train_bce_loss /= train_steps

            wandb.log({"train_loss": train_loss, "train_pu_loss": train_pu_loss, "train_bce_loss": train_bce_loss})
            
                                                            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_bce_loss = 0
                valid_pu_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        pu_loss = net(batch_features, batch_labels)
                        logits = net.logits(batch_features)

                        batch_labels = (batch_labels == 1).float()
                        bce_loss = bce(logits, batch_labels)
                        
                        valid_bce_loss += bce_loss.detach().item()
                        valid_pu_loss += pu_loss.detach().item()
                        
                        logits = net.predict(batch_features)
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_pu_loss /= valid_steps
                valid_bce_loss /= valid_steps
                #roc_auc = compute_roc(valid_labels, preds)
                fmax = compute_fmax(valid_labels, preds)
                wandb.log({"valid_pu_loss": valid_pu_loss, "valid_bce_loss": valid_bce_loss,  "valid_fmax": fmax})
                                                        
            # if valid_loss < best_loss and epoch > 1:
            if fmax > best_fmax:
                best_fmax = fmax
                # best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)
                curr_tolerance = tolerance
            else:
                curr_tolerance -= 1

            if curr_tolerance == 0:
                print('Early stopping')
                break

            
            
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file, map_location="cpu"))
    net = net.to(device)
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_loss = net(batch_features, batch_labels)
                logits = net.logits(batch_features)

                batch_labels = (batch_labels == 1).float()
                bce_loss = bce(logits, batch_labels)
                batch_loss += bce_loss
                
                test_loss += batch_loss.detach().cpu().item()
                logits = net.predict(batch_features)
                preds.append(logits.detach().cpu().numpy())
            test_loss /= test_steps
            preds = np.concatenate(preds)
            roc_auc = 0 # compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

    indexed_preds = [(i, preds[i]) for i in range(len(preds))]
    
    with get_context("spawn").Pool(30) as p:
        results = []
        with tqdm(total=len(preds)) as pbar:
            for output in p.imap_unordered(partial(propagate_annots, go=go, terms_dict=terms_dict), indexed_preds, chunksize=200):
                results.append(output)
                pbar.update()
        
        unordered_preds = [pred for pred in results]
        ordered_preds = sorted(unordered_preds, key=lambda x: x[0])
        preds = [pred[1] for pred in ordered_preds]
        
    test_df['preds'] = preds

    test_df.to_pickle(out_file)

    test(data_root, ont, model_name, run, combine, alpha_test, False, wandb_logger)
    wandb.finish()


    
def propagate_annots(preds, go, terms_dict):
    idx, preds = preds
    prop_annots = {}
    for go_id, j in terms_dict.items():
        score = preds[j]
        for sup_go in go.get_ancestors(go_id):
            if sup_go in prop_annots:
                prop_annots[sup_go] = max(prop_annots[sup_go], score)
            else:
                prop_annots[sup_go] = score
    for go_id, score in prop_annots.items():
        if go_id in terms_dict:
            preds[terms_dict[go_id]] = score
    return idx, preds


    
def compute_roc(labels, preds):
    # change labels -1 to 0
    
    labels[labels == -1] = 0

    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    labels[labels == -1] = 0
    precisions, recalls, thresholds = precision_recall_curve(labels.flatten(), preds.flatten())
    fmax = round(np.max(2 * (precisions * recalls) / (precisions + recalls + 1e-10)), 3)
    return fmax

def load_data(data_root, ont, go):
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')
                        
    train_data = get_data(train_df, terms_dict, go, data_root)
    valid_data = get_data(valid_df, terms_dict, go, data_root)
    test_data = get_data(test_df, terms_dict, go, data_root)

    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict, go_ont, data_root="data"):
    data = th.zeros((len(df), 5120), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)

    terms_count = {term: 0 for term in terms_dict.keys()}

    children = dict()
    
    for i, row in enumerate(df.itertuples()):
        data[i, :] = th.FloatTensor(row.esm2)
        if not hasattr(row, 'prop_annotations'):
            continue
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
                terms_count[go_id] += 1

        # for go_id in row.neg_annotations:
            # if go_id in terms_dict:
                # g_id = terms_dict[go_id]
                # labels[i, g_id] = -1

                # if go_id in children:
                    # descendants = children[go_id]
                # else:
                    # descendants = go_ont.get_term_set(go_id)
                    # children[go_id] = descendants
                    
                # neg_idx = [terms_dict[go] for go in descendants if go in terms_dict]
                # labels[i, neg_idx] = -1
                
                
        
    go_terms = set(terms_dict.keys())
    # Loading negatives from GOA
    goa_negs_file = f"{data_root}/goa_negative_data.txt"
    negs = set()
    with open(goa_negs_file) as f:
        for line in f:
            prot, go = line.strip().split("\t")
            negs.add((prot, go))

    # Adding InterPro negatives
    # interpro_gos = pd.read_pickle(f"{data_root}/interpro_gos.pkl")
    # ipr_gos = set(interpro_gos["gos"].values.flatten())

    
    
    
    for i, row in tqdm(enumerate(df.itertuples()), total=len(df), desc="Getting data"):
        neg_gos = set()
        for prot_ac in row.accessions:
            neg_gos.update([go for go in terms_dict if (prot_ac, go) in negs])

        # ipr2go_pos = row.interpro2go
        # ipr2go_neg = ipr_gos - ipr2go_pos
        # ipr2go_neg = ipr2go_neg & go_terms
        # neg_gos.update(ipr2go_neg)
            
        all_negs = set()
        for neg in neg_gos:
            all_negs.add(neg)
            continue
            if neg in children:
                descendants = children[neg]
            else:
                descendants = go_ont.get_term_set(neg)
                children[neg] = descendants
            # print(descendants)
            all_negs.update(descendants)

        neg_idxs = [terms_dict[go] for go in all_negs if go in terms_dict] 
         
        labels[i][neg_idxs] = -1

    num_negs = (labels == -1).sum()

    num_pos = (labels == 1).sum()
    num_unlabeled = (labels == 0).sum()

    print(f"Num pos: {num_pos}, num negs: {num_negs}, num unlabeled: {num_unlabeled}")
    
    print(f"Avg number of negatives {num_negs / len(df)}")
    
    return data, labels, terms_count


if __name__ == '__main__':
    main()

    
    
