import click as ck
import pandas as pd
import pickle as pkl
from utils import Ontology
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
import os
from tqdm import tqdm
import math
import time


from evaluate_new import test
from mowl.utils.random import seed_everything

from clearml import Task

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



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
        #net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)
    
class DeepGOPU(nn.Module):
    def __init__(self, nb_gos, prior, gamma, loss_type, terms_count, device = "cuda"):
        super().__init__()
        self.nb_gos = nb_gos
        self.prior = prior
        self.gamma = gamma
        self.dgpro = DGPROModel(nb_gos)
        self.loss_type = loss_type
        self.device = device
        
        max_count = max(terms_count.values())
        print(f"max_count: {max_count}")
        # self.priors = [self.prior*x for x in terms_count.values()]
        self.priors = [min(x/max_count, self.prior) for x in terms_count.values()]
        self.priors = th.tensor(self.priors, dtype=th.float32, requires_grad=False).to(device)
        self.weights = [1/max_count for x in terms_count.values()]
        self.weights = [1 for x in terms_count.values()]
        self.weights = th.tensor(self.weights, dtype=th.float32, requires_grad=False).to(device)
        
        
    def pu_loss(self, data, labels):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        p_above = - (F.logsigmoid(preds)*pos_label).sum() / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum() / pos_label.sum()
        u_below = - (F.logsigmoid(-preds)*unl_label).sum() / unl_label.sum()

        
        # print(f"p_above: {p_above}, p_below: {p_below}, u_below: {u_below}")
        
        loss = self.prior * p_above + th.relu(u_below - self.prior*p_below  + self.prior/2)
        return loss


    def pu_loss_multi(self, data, labels, p=1):
        preds = self.dgpro(data)

        pos_label = (labels == 1).float()
        unl_label = (labels != 1).float()

        unl_label = unl_label * (th.rand(unl_label.shape).to(self.device) < p).float()
        
        p_above = - (F.logsigmoid(preds)*pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-preds)*pos_label).sum(dim=0) / pos_label.sum()
        u_below = - (F.logsigmoid(-preds)*unl_label).sum(dim=0) / unl_label.sum()

        loss = self.priors * p_above + th.relu(u_below - self.priors*p_below + self.priors/2)
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
                                              (1 - gamma) * (u_below - self.prior * p_below + self.prior / 2))
        
        return loss

    def pun_loss_multi(self, data, labels, w_labels, p=1):
        pred = self.dgpro(data)

        p = min(p, 1)
        threshold = 1 - p
        pos_label = (labels == 1).float()
        not_pos_label = (labels != 1).float()
        unl_label = (w_labels >= -threshold).float() * not_pos_label
        neg_label = (w_labels < -threshold).float() + (labels == -1).float()

        #set unl_label values to False with probability p https://arxiv.org/pdf/2308.00279.pdf
        # unl_label = unl_label * (th.rand(unl_label.shape).to(self.device) < p).float()

        # set unl_label to negatives based on esm similarity distance
                
        
        p_above = - (F.logsigmoid(pred) * pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-pred) * pos_label).sum(dim=0) / pos_label.sum()

        if unl_label.sum() > 0:
            u_below = - (F.logsigmoid(-pred) * unl_label).sum(dim=0) / unl_label.sum()
        else:
            u_below = 0
            
        if neg_label.sum() > 0:
            n_below = - (F.logsigmoid(-pred) * neg_label).sum(dim=0) / neg_label.sum()
            gamma = self.gamma#0.01 #self.gamma 
        else:
            n_below = 0
            gamma = 0

        margin = 0 #- self.priors / 16
        loss = self.priors * p_above + th.relu(gamma * (1 - self.priors) * n_below +
                                              (1 - gamma) * (u_below - self.priors * p_below + margin))
        loss = self.weights * loss
        loss = loss.sum()
        assert loss >= 0, f"loss: {loss}"
        return loss


                                                                 
    
    def forward(self, data, labels, w_labels, p=1):
        # return self.pu_ranking_loss(data, labels)
        if self.loss_type == 'pu':
            return self.pu_loss(data, labels)
        if self.loss_type == 'pu_multi':
            return self.pu_loss_multi(data, labels)
        elif self.loss_type == "pun":
            return self.pun_loss(data, labels)
        elif self.loss_type == "pun_multi":
            return self.pun_loss_multi(data, labels, w_labels, p=p)
        else:
            raise NotImplementedError

    def logits(self, data):
        return self.dgpro(data)
    
    def predict(self, data):
        return th.sigmoid(self.dgpro(data))
 

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model-name', '-mn', default='dgpu_sample_prior',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=256,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--prior', '-p', default=1e-4,
    help='Prior')
@ck.option("--gamma", '-g', default = 0.5)
@ck.option(
    '--probability', '-prob', default=0.0,
    help='Initial probability of chosing unlabeled samples')
@ck.option("--probability-rate", '-prate', default = 0.01)
@ck.option("--alpha", '-a', default = 0.5, help="Weight of the unlabeled loss")
@ck.option('--loss_type', '-loss', default='pu', type=ck.Choice(['pu', 'pun', 'pu_multi', 'pun_multi']))
@ck.option('--max_lr', '-lr', default=1e-4)
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option("--alpha-test", "-at", default = 0.5)
@ck.option("--combine", "-c", is_flag=True)
@ck.option(
    '--device', '-d', default='cuda',
    help='Device')
def main(data_root, ont, model_name, batch_size, epochs, prior, gamma, probability, probability_rate, alpha, loss_type, max_lr, load, alpha_test, combine, device):

    
    
    log_file = f"result_{ont}.log"
    params = f"ont: {ont},"
    params += f" batch_size: {batch_size},"
    params += f" prior: {prior},"
    params += f" gamma: {gamma},"
    params += f" probability: {probability},"
    params += f" p_rate: {probability_rate},"
    params += f" alpha: {alpha},"

    with open(log_file, "a") as f:
        f.write(params + "\n")

    # cc best params: alpha 0.1, prob 0, p_rate 0.01
    
    seed_everything(0)
    go_file = f'{data_root}/go-basic.obo'
    model_name = f"{model_name}_prob{probability}_alpha{alpha}_gamma{gamma}"
    model_file = f'{data_root}/{ont}/{model_name}.th'
    out_file = f'{data_root}/{ont}/predictions_{model_name}.pkl'


    task = Task.init(project_name="deepgopu-ibex", task_name=f"{ont}_{model_name}", reuse_last_task_id=False)
    cml_logger = task.get_logger()

    
    go = Ontology(go_file, with_rels=True)
    terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, go)

    n_terms = len(terms_dict)

    
        
    train_features, train_labels, train_w_labels, terms_count = train_data
    valid_features, valid_labels, valid_w_labels, _ = valid_data
    test_features, test_labels, test_w_labels, _ = test_data


    net = DeepGOPU(n_terms, prior, gamma, loss_type, terms_count).to(device)
    

    train_loader = FastTensorDataLoader(
        train_features, train_labels, train_w_labels, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        valid_features, valid_labels, valid_w_labels, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        test_features, test_labels, test_w_labels, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    print('Labels', np.sum(valid_labels == 1))


 
    
    bce = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = th.optim.Adam(net.parameters(), lr=max_lr)
    scheduler = MultiStepLR(optimizer, milestones=[1, 3,], gamma=0.1)
    min_lr = max_lr /100
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=20, cycle_momentum=False)
    best_loss = 10000.0
    best_fmax = 0.0
    tolerance = 5
    curr_tolerance = tolerance
        
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            probability += probability_rate
            net.train()
            train_loss = 0
            train_pu_loss = 0
            train_bce_loss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels , batch_w_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_w_labels = batch_w_labels.to(device)
                    
                    pu_loss = net(batch_features, batch_labels, batch_w_labels, p=probability)
                    logits = net.logits(batch_features)

                    pos_labels = (batch_labels == 1)
                    # unl_labels = (batch_labels != 1)
                    # neg_labels = (batch_labels == -1)

                    bce_loss = bce(logits, pos_labels.float()).mean()
                    # w_bce_loss = (bce_loss[pos_labels]).mean() + (bce_loss[unl_labels] * batch_w_labels[unl_labels]).mean()

                    
                    
                    loss = alpha*pu_loss + (1-alpha)*bce_loss

                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    train_pu_loss += pu_loss.detach().item()
                    train_bce_loss += bce_loss.detach().item()
                    
                    
            train_loss /= train_steps
            train_pu_loss /= train_steps
            train_bce_loss /= train_steps


            if cml_logger:
                cml_logger.report_scalar(title="Loss evolution", series="PU Loss", iteration=epoch+1, value=train_pu_loss)
                cml_logger.report_scalar(title="Loss evolution", series="BCE Loss", iteration=epoch+1, value=train_bce_loss)
                cml_logger.report_scalar(title="Loss evolution", series="Total Loss", iteration=epoch+1, value=train_loss)
            
            
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels, batch_w_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        batch_w_labels = batch_w_labels.to(device)
                        batch_loss = net(batch_features, batch_labels, batch_w_labels)
                        logits = net.logits(batch_features)

                        batch_labels = (batch_labels == 1).float()
                        bce_loss = bce(logits, batch_labels).mean()
                        batch_loss += bce_loss
                        
                        valid_loss += batch_loss.detach().item()
                        logits = net.predict(batch_features)
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                #roc_auc = compute_roc(valid_labels, preds)
                fmax = compute_fmax(valid_labels, preds)

                if cml_logger:
                    cml_logger.report_scalar(title="Valid FMax evolution", series="FMax", iteration=epoch+1, value=fmax)
                    cml_logger.report_scalar(title="Loss evolution", series="Valid Loss", iteration=epoch+1, value=valid_loss)

                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, FMax - {fmax}')
                
            #if valid_loss < best_loss and epoch > 1:
            if fmax > best_fmax:
                best_fmax = fmax
                #best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)
                curr_tolerance = tolerance
            else:
                curr_tolerance -= 1

            if curr_tolerance == 0:
                print('Early stopping')
                break

            scheduler.step()
            
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
            for batch_features, batch_labels, batch_w_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_w_labels = batch_w_labels.to(device)
                batch_loss = net(batch_features, batch_labels, batch_w_labels)
                logits = net.logits(batch_features)

                batch_labels = (batch_labels == 1).float()
                bce_loss = bce(logits, batch_labels).mean()
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


    test(data_root, ont, model_name, combine, alpha_test, False, cml_logger)

    cml_logger.flush()
    

    
    
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
                        
    train_data = get_data(train_df, terms_dict, go, ont, "train", data_root = data_root)
    valid_data = get_data(valid_df, terms_dict, go, ont, "valid", data_root = data_root)
    test_data = get_data(test_df, terms_dict, go, ont, "test", data_root = data_root)

    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict, go_ont, ont, subset, data_root = "data"):
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

    print(f"Avg number of negatives {num_negs / len(df)}")

    # computes matrix of esm-distance

    w_labels_file = f"{data_root}/{ont}/{subset}_weighted_labels.pkl"

    if not os.path.exists(w_labels_file):
    
        device = "cuda"
        logger.info("Computing ESM distance matrix")
        esm_vectors = df["esm2"].values.tolist()
        esm_vectors = th.stack(esm_vectors)
        eye_matrix = th.eye(len(esm_vectors))
        dot_product_matrix = th.matmul(esm_vectors, esm_vectors.T)
        dot_product_matrix.mul_(1 - eye_matrix) # remove diagonal
        dot_product_matrix = th.softmax(dot_product_matrix, dim=1)
        dot_product_matrix.mul_(1 - eye_matrix)

    
        dot_product_matrix = dot_product_matrix.to(device)
        labels = labels
        weighted_labels = th.zeros_like(labels).to(device)
    
    
        mask_pos = (labels == 1).int()
        mask_neg = (labels == -1).int()

        # mask_pos = mask_pos.to(device)
        mask_neg = mask_neg.to(device)

        for i in tqdm(range(len(dot_product_matrix)), desc="Computing weighted labels"):
            prot_similarities = dot_product_matrix[i].unsqueeze(1)
        
            w_labels = prot_similarities * mask_neg
            w_labels = - w_labels.sum(dim=0)
        
            w_neg_labels = th.min(-mask_neg[i], w_labels)
            mask_pos_i = mask_pos[i].to(device)
            w_neg_labels *= (1 - mask_pos_i)
            weighted_labels[i] = mask_pos_i + w_labels
        
    
        # assert th.max(weighted_labels) <= 1, f"max value {th.max(weighted_labels)}"
        # assert th.min(weighted_labels) >= -1, f"min value {th.min(weighted_labels)}"

        weighted_labels = weighted_labels.cpu()

        with open(w_labels_file, "wb") as f:
            pkl.dump(weighted_labels, f)
    else:
        with open(w_labels_file, "rb") as f:
            weighted_labels = pkl.load(f)

        
    return data, labels, weighted_labels, terms_count


if __name__ == '__main__':
    main()
