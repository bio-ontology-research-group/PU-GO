import mowl
mowl.init_jvm("10g")
from mowl.base_models import EmbeddingELModel
from mowl.datasets import PathDataset
from mowl.nn import ELEmModule, ELBoxModule, BoxSquaredELModule
from mowl.utils.data import FastTensorDataLoader

import click as ck
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import numpy as np
from multiprocessing import Pool, get_context
from functools import partial
import os
from dataset import load_data, PFDataset, load_testing_data
from utils import Ontology, seed_everything
import logging
import time
import gc


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

                
        projection = []
        projection.append(MLPBlock(esm_dim, el_dim))
        projection.append(Residual(MLPBlock(el_dim, el_dim)))
        self.projection = nn.Sequential(*projection)

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
        
        class_embed = self.el_module.class_center if self.module_name == "box2el" else self.el_module.class_embed
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
            if self.module_name == "elbox":
                membership = membership + offset_embed
                    
        membership = th.sigmoid(membership)
        return membership

class ELModel(EmbeddingELModel):

    def __init__(self, data_root, ont, module, num_models, esm_dim, terms_dict, lr, cafa, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.ont = ont
        self.module = module
        self.num_models = num_models
        self.esm_dim = esm_dim
        self.terms_dict = terms_dict
        self.lr = lr
        self.cafa = cafa
        self.test_loader = None
        self.test_df = None
        self._init_modules()
        
        go_obo_file = f"{self.data_root}/go-basic.obo"
        self.go_ont = Ontology(go_obo_file, with_rels=True)


    def _init_modules(self):
        logger.info("Initializing modules...")
        nb_classes = len(self.dataset.classes)
        nb_roles = len(self.dataset.object_properties)
        has_func_owl = self.dataset.object_properties.to_dict()["http://mowl/has_function"]
        has_func_id = self.dataset.object_properties.to_index_dict()[has_func_owl]
        modules = []
        for i in range(self.num_models):
            module = PFModule(self.module,
                              self.esm_dim,
                              self.embed_dim,
                              nb_classes,
                              nb_roles,
                              self.terms_dict,
                              has_func_id)
            modules.append(module)
        self.modules = nn.ModuleList(modules)
        logger.info(f"Number of models created: {len(modules)}")
            
    def load_train_data(self):
        logger.info("Loading training data...")
        start = time.time()
        terms_dict, train_data, valid_data, feature_size = load_data(self.data_root, self.ont, self.go_ont, cafa=self.cafa)
        self.terms_dict = terms_dict
        self.train_data = train_data
        self.valid_data = valid_data
        self.feature_size = feature_size

        self.train_loader = FastTensorDataLoader(*self.train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = FastTensorDataLoader(*self.valid_data, batch_size=self.batch_size, shuffle=False)
        
        el_dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}

        el_dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        total_el_dls_size = sum(el_dls_sizes.values())
        self.el_dls_weights = {gci_name: ds_size / total_el_dls_size for gci_name, ds_size in el_dls_sizes.items()}
        
        self.el_dls = {gci_name: cycle(dl) for gci_name, dl in el_dls.items()}
        logger.info(f"Dataloaders: {el_dls.keys()}")

        end = time.time()
        logger.info(f"Data loaded in {end-start:.2f} seconds")
        

    def load_test_data(self):
        logger.info("Loading testing data...")
        start = time.time()
        terms_dict, test_data, feature_size, test_df = load_testing_data(self.data_root, self.ont, self.go_ont, cafa=self.cafa)
        self.terms_dict = terms_dict
        self.test_data = test_data
        self.test_df = test_df
        self.feature_size = feature_size

        self.test_loader = FastTensorDataLoader(*self.test_data, batch_size=self.batch_size, shuffle=False)

        end = time.time()
        logger.info(f"Data loaded in {end-start:.2f} seconds")

    def train(self, epochs=None):
        self.load_train_data()
        
        
        
        tolerance = 5
        current_tolerance = tolerance

        nb_classes = len(self.dataset.classes)

        for i, module in enumerate(self.modules):
            optimizer = optim.Adam(module.parameters(), lr=self.lr)
            logger.info(f"Training model {i+1}/{len(self.modules)}")
            sub_model_filepath = self.model_filepath.replace(".pt", f"_{i+1}_of_{len(self.modules)}.pt")
            module.train()
            module = module.to(self.device)
            best_loss = float("inf")
            for epoch in tqdm(range(epochs), desc="Training..."):
                train_pf_loss = 0.0
                train_el_loss = 0.0
                for batch_features, batch_labels in self.train_loader:
                    optimizer.zero_grad()
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    membership = module.pf_forward(batch_features)
                    pf_loss = F.binary_cross_entropy(membership, batch_labels)
                    el_loss = 0
                    for gci_name, dl in self.el_dls.items():
                        el_loss += module.el_forward(next(dl).to(self.device), gci_name).mean() * self.el_dls_weights[gci_name]
                        # data = next(dl).to(self.device)
                        # pos_logits = module.el_forward(next(dl).to(self.device), gci_name).mean() * self.el_dls_weights[gci_name]
                        # neg_idxs = np.random.choice(nb_classes, size=data.shape[0])
                        # neg_idxs = th.tensor(neg_idxs).to(self.device)
                        # neg_batch = th.cat((data[:, :-1], neg_idxs.unsqueeze(1)), dim=1)
                        # neg_logits = module.el_forward(neg_batch, gci_name).mean() * self.el_dls_weights[gci_name]
                        # el_loss += -F.logsigmoid(-pos_logits + neg_logits - 1e-8).mean()

                    loss = pf_loss + el_loss
                    loss.backward()
                    optimizer.step()
                    train_pf_loss += pf_loss.item()
                    train_el_loss += el_loss.item()
                    
                train_pf_loss /= len(self.train_loader)
                train_el_loss /= len(self.train_loader)

                module.eval()
                with th.no_grad():
                    valid_loss = 0
                    preds = []
                    labels = []

                    for batch_features, batch_labels in self.valid_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        membership = module.pf_forward(batch_features)
                        pf_loss = F.binary_cross_entropy(membership, batch_labels)
                        valid_loss += pf_loss.item()


                        preds = np.concatenate([preds, membership.cpu().numpy().flatten()])
                        labels = np.concatenate([labels, batch_labels.cpu().numpy().flatten()])

                    valid_loss /= len(self.valid_loader)

                    logger.debug(f"preds.shape: {preds.shape}")
                    logger.debug(f"labels.shape: {labels.shape}")
                    roc_auc = compute_roc(labels, preds)

                    logger.info(f"Epoch: {epoch+1}/{epochs} | PF loss: {train_pf_loss:.4f} | EL loss: {train_el_loss:.4f} | Valid loss: {valid_loss:.4f} | ROC AUC: {roc_auc:.4f}")

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        logging.info("Saving model...")
                        th.save(module.state_dict(), sub_model_filepath)
                        curr_tolerance = tolerance
                    else:
                        curr_tolerance -= 1



                    if curr_tolerance == 0:
                        logger.info("Early stopping...")
                        break

            # remove everything from gpu
            del module
            del optimizer
            del train_pf_loss
            del train_el_loss
            del valid_loss
            del batch_features
            del batch_labels
            del membership
            del pf_loss
            del el_loss
            del loss
            
            th.cuda.empty_cache()
            gc.collect()
            logger.info("Model removed from GPU")
            
                                     
        
    def test(self, out_file):
        self.load_test_data()
        
        # Loading best model
        print('Loading modules from disk...')
        for i, module in enumerate(self.modules):
            logger.info(f"Loading model {i+1}/{len(self.modules)}")
            sub_model_filepath = self.model_filepath.replace(".pt", f"_{i+1}_of_{len(self.modules)}.pt")
            module.load_state_dict(th.load(sub_model_filepath))
            
        self.modules.eval()
        with th.no_grad():
            test_loss = 0
            all_preds = []
            for i, module in enumerate(self.modules):
                module = module.to(self.device)
                preds = []
                for batch_features, batch_labels in tqdm(self.test_loader):
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    logits = module.pf_forward(batch_features)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds.append(logits.detach().cpu().numpy())
                test_loss /= len(self.test_loader)
                preds = np.concatenate(preds)
                all_preds.append(preds)

                #remove module from GPU
                module = module.cpu()
                
            all_preds = np.stack(all_preds, axis=1)

            #aggregators = {"mean": np.mean, "max": np.max, "min": np.min, "median": np.median}
            aggregators = {"min": np.min}
            
            agg_preds = {agg_name: aggregator(all_preds, axis=1) for agg_name, aggregator in aggregators.items()}
                                                                                                                                                        
            #roc_auc = compute_roc(test_labels, preds)
        #print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

        #preds = list(preds)

        ancestors_dict = {}
        #Precomputing ancestors
        logger.info('Precomputing ancestors')
        for go_id in tqdm(self.terms_dict, total=len(self.terms_dict)):
            ancestors_dict[go_id] = self.go_ont.get_ancestors(go_id)

        propagate = True
        if propagate:
            for agg_name, agg_pred in agg_preds.items():

                indexed_preds = [(i, agg_pred[i]) for i in range(len(agg_pred))]


                # Propagate scores using ontology structure
                with get_context("spawn").Pool(processes=30) as p:
                    results = []
                    with tqdm(total=len(agg_pred)) as pbar:
                        for output in p.imap_unordered(partial(propagate_annots, go=self.go_ont, terms_dict=self.terms_dict, ancestors_dict=ancestors_dict), indexed_preds, chunksize=200):
                            results.append(output)
                            pbar.update()

                    unordered_preds = [pred for pred in results]
                    ordered_preds = sorted(unordered_preds, key=lambda x: x[0])
                    final_preds = [pred[1] for pred in ordered_preds]

                agg_preds[agg_name] = final_preds

            for agg_name, agg_pred in agg_preds.items():
                self.test_df[f"preds_{agg_name}"] = agg_pred
        else:
            
            for agg_name, agg_pred in agg_preds.items():
                agg_pred = [agg_pred[i] for i in range(len(agg_pred))]
                self.test_df[f"preds_{agg_name}"] = agg_pred
                
        self.test_df.to_pickle(out_file)



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

    
def propagate_annots(preds, go, terms_dict, ancestors_dict):
    idx, preds = preds
    prop_annots = {}
    for go_id, j in terms_dict.items():
        score = preds[j]
        
        for sup_go in ancestors_dict[go_id]:
           prop_annots[sup_go] = max(prop_annots.get(sup_go, 0), score)

        # ancestor_scores = np.array([prop_annots.get(sup_go, 0) for sup_go in ancestors_dict[go_id]])
        # new_ancestor_scores = np.maximum(ancestor_scores, score)
        # for sup_go, new_score in zip(ancestors_dict[go_id], new_ancestor_scores):
            # prop_annots[sup_go] = new_score

    terms_idxs = [terms_dict[go_id] for go_id in prop_annots if go_id in terms_dict]
    scores = [prop_annots[go_id] for go_id in prop_annots if go_id in terms_dict]
    #assert len(terms_idxs) == len(scores)
    preds[terms_idxs] = scores
    #for go_id, score in prop_annots.items():
    #    if go_id in terms_dict:
    #        preds[terms_dict[go_id]] = score
    return idx, preds


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return roc_auc



@ck.command()
@ck.option("--data-root", "-root", default="data/")
@ck.option("--module", type=ck.Choice(["elem", "elbox", "box2el"]), default="elem")
@ck.option("--num-models", "-nm", default=10)
@ck.option("--esm-dim", default=5120)
@ck.option("--el-dim", "-dim", default=1024) #stable with 1024
@ck.option("--ont", "-ont", type=ck.Choice(["mf", "bp", "cc"]), default="mf")
@ck.option("--batch-size", "-bs", default=128)
@ck.option("--lr", "-lr", default=5e-4)
@ck.option("--epochs", "-e", default=100)
@ck.option("--device", "-d", default="cuda")
@ck.option("--only-test", "-ot", is_flag=True)
@ck.option("--cafa", "-cafa", is_flag=True)
def main(data_root, module, num_models, esm_dim, el_dim, ont, batch_size, lr, epochs, device, only_test, cafa):
    train_file = f"{data_root}/{ont}/train.owl"
    valid_file = f"{data_root}/{ont}/valid.owl"

    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    parent_dir = os.path.dirname(data_root)
    model_filepath = f"{parent_dir}/models/{ont}_{module}.pt"
    dataset = PFDataset(train_file, valid_file)
    model = ELModel(data_root, ont, module, num_models, esm_dim, terms_dict, lr, cafa, dataset, el_dim, batch_size, device = device, model_filepath=model_filepath)
    if not only_test:
        model.train(epochs=epochs)

    predictions_file = f"{data_root}/{ont}/test_predictions_{module}.pkl"
    if cafa:
        predictions_file = f"{data_root}/{ont}/cafa_predictions_{module}.pkl"
    model.test(predictions_file)

if __name__ == "__main__":
    seed_everything(42)
    main()
