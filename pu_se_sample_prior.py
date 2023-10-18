import click as ck
import pandas as pd
from utils import Ontology
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import copy
import pickle as pkl
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
from itertools import cycle
import sys
from tqdm import tqdm
import math
import time
import os

import mowl
mowl.init_jvm("10g")
from mowl.base_models import EmbeddingELModel
from mowl.datasets import PathDataset
from mowl.nn import ELEmModule, ELBoxModule, BoxSquaredELModule

from mowl.utils.random import seed_everything
from evaluate_sem import test
import wandb
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PFDataset(PathDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                         
        self._proteins = None
        self._functions = None
        
    @property
    def functions(self):
        if self._functions is None:
            functions = set()
            for cls_str, cls_owl in self.classes.as_dict.items():
                if cls_str.startswith("http://purl.obolibrary.org/obo/GO"):
                    functions.add(cls_owl)
            self._functions = OWLClasses(functions)
        return self._functions

    @property
    def proteins(self):
        if self._proteins is None:
            proteins = set()
            for ind_str, ind_owl in self.individuals.as_dict.items():
                if ind_str.startswith("http://mowl/protein"):
                    proteins.add(ind_owl)
            self._proteins = OWLIndividuals(proteins)
        return self._proteins

    @property
    def evaluation_property(self):
        return "http://mowl/has_function"


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
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)

class PFModule(nn.Module):
    def __init__(self, el_module, esm_dim, el_dim, nb_classes, nb_roles, terms_dict, has_func_id):
        super(PFModule, self).__init__()

        self.esm_dim = esm_dim
        self.el_dim = el_dim
        self.nb_classes = nb_classes
        self.nb_roles = nb_roles
        self.terms_dict = terms_dict
        self.module_name = el_module
        self._set_module(el_module)

        self.projection = DGPROModel(nb_classes)
        
        self.terms_dict = terms_dict
        self.go_terms_ids = th.tensor([list(self.terms_dict.values())], dtype=th.long)
        self.has_func_id = th.tensor(has_func_id, dtype=th.long)

        
    def _set_module(self, module):
        if module == "elem":
            self.el_module = ELEmModule(self.nb_classes, self.nb_roles, embed_dim = self.el_dim)
        elif module == "elbox":
            self.el_module = ELBoxModule(self.nb_classes, self.nb_roles, embed_dim = self.el_dim)
        elif module == "box2el":
            self.el_module = BoxSquaredELModule(self.nb_classes, self.nb_roles, embed_dim = self.el_dim)
        else:
            raise ValueError("Unknown module: {}".format(module))


    def el_forward(self, *args, **kwargs):
        return self.el_module(*args, **kwargs)

    def pf_forward(self, features):
        el_features = self.projection(features)
        go_terms_ids = self.go_terms_ids.to(el_features.device)
        has_func_id = self.has_func_id.to(el_features.device)
        class_embed = self.el_module.bump if self.module_name == "box2el" else self.el_module.class_embed
        rel_embed = self.el_module.head_center if self.module_name == "box2el" else self.el_module.rel_embed
        go_embeds = class_embed(go_terms_ids).squeeze(0)
        has_func = rel_embed(has_func_id)
        has_func =  has_func - go_embeds if self.module_name == "box2el" else go_embeds - has_func
        membership = th.matmul(el_features, has_func.t())
        if self.module_name == "elem":
            rad_embed = self.el_module.class_rad(go_terms_ids).squeeze(0)
            rad_embed = th.abs(rad_embed).view(1, -1)
            membership = membership + rad_embed
        elif self.module_name in ["elbox", "box2el"]:
            offset_embed = self.el_module.class_offset(go_terms_ids).squeeze(0)
            offset_embed = th.abs(offset_embed).mean(dim=1).view(1, -1)
            membership = membership + offset_embed
                                            
        return membership

    
class DeepGOPU(EmbeddingELModel):
    def __init__(self, data_root, model_name, load, ont, pf_data, go_ont, el_module, num_models, max_lr, probability, probability_rate, alpha, beta, prior, gamma, loss_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.terms_dict, train_data, valid_data, test_data, self.test_df = pf_data
        self.train_features, self.train_labels, self.train_w_labels, self.terms_count = train_data
        self.valid_features, self.valid_labels, self.valid_w_labels, _ = valid_data
        self.test_features, self.test_labels, self.test_w_labels, _ = test_data
        
        self.data_root = data_root
        self.ont = ont
        self.go_ont = go_ont

        self.load_el_data()
        
        self.load = load
        self.module = el_module
        self.num_models = num_models
        self.max_lr = max_lr
        self.probability = probability
        self.probability_rate = probability_rate
        self.alpha = alpha
        self.beta = beta
        
        self.out_file = f'{data_root}/{ont}/predictions_{model_name}.pkl'

        self.nb_classes = len(self.dataset.classes)
        self.nb_roles = len(self.dataset.object_properties)
        has_func_owl = self.dataset.object_properties.to_dict()["http://mowl/has_function"]
        self.has_func_id = self.dataset.object_properties.to_index_dict()[has_func_owl]

        self.nb_gos = len(self.terms_dict)

        self.prior = prior
        self.gamma = gamma

        self.loss_type = loss_type
        
        max_count = max(self.terms_count.values())
        print(f"max_count: {max_count}")
        # self.priors = [self.prior*x for x in terms_count.values()]
        self.priors = [min(x/max_count, self.prior) for x in self.terms_count.values()]
        self.priors = th.tensor(self.priors, dtype=th.float32, requires_grad=False).to(self.device)
        self.weights = [1/max_count for x in self.terms_count.values()]
        self.weights = [1 for x in self.terms_count.values()]
        self.weights = th.tensor(self.weights, dtype=th.float32, requires_grad=False).to(self.device)
        
        self._init_modules()

    def _init_modules(self):
        logger.info("Initializing modules...")
        modules = []
        for i in range(self.num_models):
            module = PFModule(self.module,
                              5120,
                              self.embed_dim,
                              self.nb_classes,
                              self.nb_roles,
                              self.terms_dict,
                              self.has_func_id)
            modules.append(module)
        self.modules = nn.ModuleList(modules)
        logger.info(f"Number of models created: {len(modules)}")
    

    def pun_loss_multi(self, pred, labels, w_labels, p=1):

        p = min(p, 1)
        threshold = 1 - p
        
        pos_label = (labels == 1).float()
        not_pos_label = (labels != 1).float()

        unl_label = (w_labels >= - threshold).float() * not_pos_label
        neg_label = (w_labels < - threshold).float() + (labels == -1).float()
                
        p_above = - (F.logsigmoid(pred) * pos_label).sum(dim=0) / pos_label.sum()
        p_below = - (F.logsigmoid(-pred) * pos_label).sum(dim=0) / pos_label.sum()

        if unl_label.sum() > 0:
            u_below = - (F.logsigmoid(-pred) * unl_label).sum(dim=0) / unl_label.sum()
        else:
            u_below = 0
            
        if neg_label.sum() > 0:
            n_below = - (F.logsigmoid(-pred) * neg_label).sum(dim=0) / neg_label.sum()
            gamma = self.gamma #0.01 #self.gamma 
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

    
    def forward(self, data, labels, p=1):
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
 
    def load_el_data(self):
        logger.info("Loading EL data...")
        start = time.time()
                                                                
        
        
        el_dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}

        el_dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        total_el_dls_size = sum(el_dls_sizes.values())
        self.el_dls_weights = {gci_name: ds_size / total_el_dls_size for gci_name, ds_size in el_dls_sizes.items()}
        
        self.el_dls = {gci_name: cycle(dl) for gci_name, dl in el_dls.items()}
        logger.info(f"Dataloaders: {el_dls.keys()}")

        end = time.time()
        logger.info(f"Data loaded in {end-start:.2f} seconds")

 
    def train(self, epochs = 100):
        
        bce = nn.BCEWithLogitsLoss()
        
        train_loader = FastTensorDataLoader(self.train_features, self.train_labels, self.train_w_labels, batch_size=self.batch_size, shuffle=True)
        valid_loader = FastTensorDataLoader(self.valid_features, self.valid_labels, self.valid_w_labels, batch_size=self.batch_size, shuffle=False)
        
        
        if not self.load:

            for i, module in enumerate(self.modules):
                logger.info(f'Training model {i+1}/{len(self.modules)}')

                optimizer = th.optim.Adam(module.parameters(), lr=self.max_lr)
                min_lr = self.max_lr /100
                scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=self.max_lr, step_size_up=20, cycle_momentum=False)
                
                tolerance = 5
                curr_tolerance = tolerance

                sub_model_file = self.model_filepath.replace(".th", f"_{i+1}_of_{len(self.modules)}.th")
                module.train()
                module = module.to(self.device)
                best_loss = 10000.0
                best_fmax = 0.0
                for epoch in range(epochs):
                    self.probability += self.probability_rate
                    module.train()
                    train_pu_loss = 0
                    train_el_loss = 0
                    train_bce_loss = 0
                    train_steps = int(math.ceil(len(self.train_labels) / self.batch_size))
                    with ck.progressbar(length=train_steps, show_pos=True) as bar:
                        for batch_features, batch_labels, batch_w_labels in train_loader:
                            bar.update(1)
                            batch_features = batch_features.to(self.device)
                            batch_labels = batch_labels.to(self.device)
                            batch_w_labels = batch_w_labels.to(self.device)
                            
                            mem_logits = module.pf_forward(batch_features)
                            
                            pu_loss = self.pun_loss_multi(mem_logits, batch_labels, batch_w_labels, p=self.probability)
                            batch_labels = (batch_labels == 1).float()
                            bce_loss = bce(mem_logits, batch_labels)
                            
                            el_loss = 0
                            for gci_name, dl in self.el_dls.items():
                                
                                gci_batch = next(dl).to(self.device)
                                pos_gci = module.el_forward(gci_batch, gci_name).mean()
                                neg_idxs = np.random.choice(self.nb_classes, size=len(gci_batch), replace=True)
                                neg_batch = th.tensor(neg_idxs, dtype=th.long, device=self.device)
                                neg_data = th.cat((gci_batch[:, :2], neg_batch.unsqueeze(1)), dim=1)
                                neg_gci = module.el_forward(neg_data, gci_name).mean()# * el_dls_weights[gci_name]
                                margin = 0 
                                el_loss += -F.logsigmoid(-pos_gci + neg_gci - margin).mean()

                            loss = self.beta * el_loss + (1-self.beta)*((1-self.alpha)*bce_loss + self.alpha * pu_loss)
                                                        
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_pu_loss += pu_loss.item()
                            train_el_loss += el_loss.item()
                            train_bce_loss += bce_loss.item()
                    train_pu_loss /= train_steps
                    train_el_loss /= train_steps
                    train_bce_loss /= train_steps

                    wandb.log({"train_pu_loss": train_pu_loss, "train_el_loss": train_el_loss, "train_bce_loss": train_bce_loss})
                    
                    print('Validation')
                    module.eval()
                    with th.no_grad():
                        valid_steps = int(math.ceil(len(self.valid_labels) / self.batch_size))
                        valid_loss = 0
                        preds = []
                        with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                            for batch_features, batch_labels, batch_w_labels in valid_loader:
                                bar.update(1)
                                batch_features = batch_features.to(self.device)
                                batch_labels = batch_labels.to(self.device)

                                mem_logits = th.sigmoid(module.pf_forward(batch_features))
                                bce_loss = bce(mem_logits, batch_labels)

                                valid_loss += bce_loss.detach().item()
                                preds = np.append(preds, mem_logits.detach().cpu().numpy())
                                
                        valid_loss /= valid_steps
                        valid_fmax = compute_fmax(self.valid_labels, preds)
                        wandb.log({"valid_loss": valid_loss, "valid fmax": valid_fmax})
                        print(f'Epoch {epoch}: PF Loss - {train_pu_loss:.6f}, EL Loss - {train_el_loss:.6f}, BCE Loss - {train_bce_loss:.6f}, Valid loss - {valid_loss:.6f}, Valid Fmax - {valid_fmax:.6f}')

                    if valid_fmax > best_fmax:
                        best_fmax = valid_fmax
                        print('Saving model')
                        th.save(module.state_dict(), sub_model_file)
                        curr_tolerance = tolerance
                    else:
                        curr_tolerance -= 1

                    if curr_tolerance == 0:
                        print('Early stopping')
                        break

                    scheduler.step()

    def predict(self):
        test_loader = FastTensorDataLoader(self.test_features, self.test_labels, self.test_w_labels, batch_size=self.batch_size, shuffle=False)
        logger.info("Loading models from disk...")

        bce = nn.BCEWithLogitsLoss()
        
        for i, module in enumerate(self.modules):
            logger.info(f"Loading model {i+1}/{len(self.modules)}")
            sub_model_filepath = self.model_filepath.replace(".th", f"_{i+1}_of_{len(self.modules)}.th")
            module.load_state_dict(th.load(sub_model_filepath))

        self.modules.eval()
                                
        with th.no_grad():
            test_steps = int(math.ceil(len(self.test_labels) / self.batch_size))
            test_loss = 0
            all_preds = []

            for i, module in enumerate(self.modules):
                module = module.to(self.device)
                preds = []
                with ck.progressbar(length=test_steps, show_pos=True) as bar:
                    for batch_features, batch_labels, batch_w_labels in test_loader:
                        bar.update(1)
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        logits = th.sigmoid(module.pf_forward(batch_features))
                        batch_loss = bce(logits, batch_labels)
                        test_loss += batch_loss.detach().cpu().item()
                        
                        preds.append(logits.detach().cpu().numpy())
                    test_loss /= test_steps
                    preds = np.concatenate(preds)
                    roc_auc = 0 # compute_roc(self.test_labels, preds)
                    print(f'Test Loss {i} - {test_loss}, AUC - {roc_auc}')

                    
                all_preds.append(preds)
                module = module.cpu()

            all_preds = np.stack(all_preds, axis=1)
            aggregators = {"mean": np.mean, "max": np.max, "min": np.min, "median": np.median}
            # aggregators = {"mean": np.mean}

            agg_preds = {agg_name: aggregator(all_preds, axis=1) for agg_name, aggregator in aggregators.items()}



        for agg_name, agg_pred in agg_preds.items():
            
            indexed_preds = [(i, agg_pred[i]) for i in range(len(agg_pred))]

            with get_context("spawn").Pool(10) as p:
                results = []
                with tqdm(total=len(agg_pred)) as pbar:
                    for output in p.imap_unordered(partial(propagate_annots, go=self.go_ont, terms_dict=self.terms_dict), indexed_preds, chunksize=200):
                        results.append(output)
                        pbar.update()

            unordered_preds = [pred for pred in results]
            ordered_preds = sorted(unordered_preds, key=lambda x: x[0])
            preds = [pred[1] for pred in ordered_preds]

            agg_preds[agg_name] = preds

        for agg_name, preds in agg_preds.items():
            self.test_df[f'preds_{agg_name}'] = preds
            
        self.test_df.to_pickle(self.out_file)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    "--el-model", "-el", default="elem", type=ck.Choice(["elem", "elbox", "box2el"]),
    help="Semantic method")
@ck.option(
    "--num-models", "-nmodels", default = 1
    )
@ck.option(
    '--model-name', '-mn', default='dgpu-pu-sample-prior',
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
@ck.option("--gamma", '-g', default = 0.01)
@ck.option('--probability', '-prob', default=0.0, help='Initial probability of chosing unlabeled samples')
@ck.option("--probability-rate", '-prate', default = 0.01)
@ck.option("--alpha", '-a', default = 0.5, help="Weight of the BCE loss")
@ck.option("--beta", '-b', default = 0.1, help="Weight of the EL loss. Using best one from semantic baseline")
@ck.option('--loss_type', '-loss', default='pu', type=ck.Choice(['pu', 'pun', 'pu_multi', 'pun_multi']))
@ck.option('--max_lr', '-lr', default=1e-4)
@ck.option('--alpha_test', '-at', default=0.5)
@ck.option("--combine", '-c', is_flag=True)
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda',
    help='Device')
def main(data_root, ont, el_model, num_models, model_name, batch_size, epochs, prior, gamma, probability, probability_rate, alpha, beta, loss_type, max_lr, alpha_test, combine, load, device):

    
    train_file = f'{data_root}/{ont}/train_normalized.owl'
    valid_file = f'{data_root}/{ont}/valid.owl'
    test_file = f'{data_root}/{ont}/test.owl'
    
    log_file = f"result_sem_{ont}.log"
    params = f"ont: {ont},"
    params += f"el_model: {el_model},"
    params += f" batch_size: {batch_size},"
    params += f" prior: {prior},"
    params += f" gamma: {gamma},"
    params += f" probability: {probability},"
    params += f" p_rate: {probability_rate},"
    params += f" alpha: {alpha},"
    params += f" beta: {beta},"

    
    with open(log_file, "a") as f:
        f.write(params + "\n")

    # cc best params: alpha 0.1, prob 0, p_rate 0.01
    
    seed_everything(0)

    
    
    go_file = f'{data_root}/go-basic.obo'
    go = Ontology(go_file, with_rels=True)
    pf_data = load_data(data_root, ont, go)

    model_name = f"{model_name}_{ont}_{el_model}_prob{probability}_alpha{alpha}_gamma{gamma}"

    sim = "_sim" if "sim" in data_root else ""
    wandb_logger = wandb.init(project="dgpu_se" + sim, name=model_name, group = ont +"_sample_prior")
    wandb.config.update({"root": data_root,
                         "ont": ont,
                         "el_model": el_model,
                         "num_models": num_models,
                         "model_name": model_name,
                         "batch_size": batch_size,
                         "prior": prior,
                         "gamma": gamma,
                         "probability": probability,
                         "p_rate": probability_rate,
                         "alpha": alpha,
                         "beta": beta,
                         "loss_type": loss_type,
                         "max_lr": max_lr})
    
    model_file = f'{data_root}/{ont}/{model_name}.th'
    logger.info(f"Creating mOWL dataset")
    start = time.time()
    dataset = PFDataset(train_file, valid_file, test_file)
    end = time.time()
    logger.info(f"Dataset created in {end - start} seconds")
    
    el_dim = 2048

    

    model = DeepGOPU(data_root, model_name, load, ont, pf_data, go, el_model, num_models, max_lr, probability, probability_rate, alpha, beta, prior, gamma, loss_type, dataset, el_dim, batch_size, device = device, model_filepath = model_file)

    wandb.watch(model.modules)
    
    model.train(epochs = epochs)
    model.predict()

    for agg in ["min", "max", "mean", "median"]:
        test(data_root, ont, model_name, False, alpha_test, False, agg,  wandb_logger)
                
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
    logger.info(f"Loading protein function data...")
    start = time.time()
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))

    logger.info(f"Reading pickle files")
    start = time.time()
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')
    logger.info(f"Read pickle files in {time.time() - start} seconds")
    
    train_data = get_data(train_df, terms_dict, go, ont, "train", data_root = data_root)
    valid_data = get_data(valid_df, terms_dict, go, ont, "valid", data_root = data_root)
    test_data = get_data(test_df, terms_dict, go, ont, "test", data_root = data_root)
    end = time.time()
    logger.info(f"Data loaded in {end - start} seconds")
    
    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict, go_ont, ont, subset, data_root="data/"):
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
