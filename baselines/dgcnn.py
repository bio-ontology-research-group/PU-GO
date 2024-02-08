import click as ck
import pandas as pd
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR
from utils import Ontology
from multiprocessing import Pool, get_context
from functools import partial
from tqdm import tqdm
import wandb
from evaluate_new import test

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data-sim',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot', 'valid']),
    help='Test data set name')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=30,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(data_root, ont, test_data_name, batch_size, epochs, load, device):

    combine = False
    run = 0

    wandb_logger = wandb.init(project="final-dgpu-similarity-based", name= ont, group="dgcnn")
    model_name = f"deepgocnn_{ont}"

    
    go_file = f'{data_root}/go-basic.obo'
    model_file = f'{data_root}/{ont}/deepgocnn.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgocnn_{ont}_{run}.pkl'
    
    go = Ontology(go_file, with_rels=True)
    loss_func = nn.BCELoss()
    test_data_file = f'{test_data_name}_data.pkl'
    terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file, test_data_file=test_data_file)
    n_terms = len(terms_dict)
    
    net = DGCNNModel(n_terms, device).to(device)
    
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[1, 3,], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds.append(logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                preds = np.concatenate(preds)
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()
            
    
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
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
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

    preds = list(preds)


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

    # Propagate scores using ontology structure
    # with Pool(32) as p:
        # preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)

    test_df['preds'] = preds
    test_df.to_pickle(out_file)

    test(data_root, ont, model_name, run, combine, 0.5, False, wandb_logger)

class DGCNNModel(nn.Module):

    def __init__(self, nb_gos, device, nb_filters=512, max_kernel=129, hidden_dim=1024):
        super().__init__()
        self.nb_gos = nb_gos
        # DeepGOCNN
        kernels = range(8, max_kernel, 8)
        convs = []
        for kernel in kernels:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(22, nb_filters, kernel, device=device),
                    nn.MaxPool1d(MAXLEN - kernel + 1)
                ))
        self.convs = nn.ModuleList(convs)
        self.fc1 = nn.Linear(len(kernels) * nb_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, nb_gos)
        
    def deepgocnn(self, proteins):
        n = proteins.shape[0]
        output = []
        for conv in self.convs:
            output.append(conv(proteins))
        output = th.cat(output, dim=1)
        output = th.relu(self.fc1(output.view(n, -1)))
        output = th.sigmoid(self.fc2(output))
        return output
    
    def forward(self, proteins):
        return self.deepgocnn(proteins)


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
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def load_data(data_root, ont, terms_file, test_data_file='test_data.pkl'):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)

    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict):
    data = th.zeros((len(df), 22, MAXLEN), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        seq = row.sequences
        seq = th.FloatTensor(to_onehot(seq))
        data[i, :, :] = seq
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
