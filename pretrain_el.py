import mowl
mowl.init_jvm("10g")
from mowl.nn import ELEmModule
from mowl.datasets import PathDataset
from mowl.datasets import ELDataset

from itertools import cycle

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pickle as pkl

device = "cuda"
margin = 0.1
go_path = f"data-sim/go-plus-el.owl"
dataset = PathDataset(go_path)
el_dataset = ELDataset(dataset.ontology)
nb_classes = len(el_dataset.class_index_dict)
nb_roles = len(el_dataset.object_property_index_dict)

training_dataloaders = {k: DataLoader(v, batch_size = 256) for k,v in el_dataset.get_gci_datasets().items()}
training_dataloaders = {gci_name: cycle(dl) for gci_name, dl in training_dataloaders.items() if len(dl) > 0}

el_dls_sizes = {gci_name: len(ds) for gci_name, ds in el_dataset.get_gci_datasets().items() if len(ds) > 0}
total_el_dls_size = sum(el_dls_sizes.values())
el_dls_weights = {gci_name: ds_size / total_el_dls_size for gci_name, ds_size in el_dls_sizes.items()}

model = ELEmModule(nb_classes, nb_roles, embed_dim=1024).to(device)
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

last_loss = float("inf")

train = False
if train:
    for step in range(50000):
        model.train()
        train_loss = 0
        el_loss = 0
        for gci_name, gci_dl in training_dataloaders.items():
            gci_batch = next(gci_dl).to(device)
            pos_gci = model(gci_batch, gci_name).mean() * el_dls_weights[gci_name]
            neg_idxs = np.random.choice(nb_classes, size=len(gci_batch), replace=True)
            neg_batch = th.tensor(neg_idxs, dtype=th.long, device=device)
            neg_data = th.cat((gci_batch[:, :-1], neg_batch.unsqueeze(1)), dim=1)
            neg_gci = model(neg_data, gci_name).mean() * el_dls_weights[gci_name]
            el_loss += -F.logsigmoid(-pos_gci + neg_gci - margin).mean()
        optimizer.zero_grad()
        el_loss.backward()
        optimizer.step()

        train_loss += el_loss.item()



        if step % 1000 == 0:
            print(f"Step {step} - EL Loss: {train_loss:.4f}")
            if train_loss < last_loss:
                last_loss = train_loss
                th.save(model.state_dict(), f"models-sim/el.pt")

model.load_state_dict(th.load(f"models-sim/el.pt"))

class_centers = model.class_embed
class_radius = model.class_rad

embeds = dict()
for go_name, go_id in el_dataset.class_index_dict.items():
    go_id = th.tensor(go_id, dtype=th.long, device=device)
    center = class_centers(go_id).squeeze()
    radius = class_radius(go_id).squeeze()
    embed = th.cat((center, radius.unsqueeze(0)))
    assert embed.shape == (1025,), f"Embed shape is {embed.shape}"
    embeds[go_name] = embed


pkl.dump(embeds, open("models-sim/el.pkl", "wb"))
    
