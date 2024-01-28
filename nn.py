import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
    
class PUModel(nn.Module):
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
        self.priors = [self.prior*x/max_count for x in terms_count.values()]
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

    def forward(self, data, labels):
        if self.loss_type == 'pu':
            return self.pu_loss(data, labels)
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

