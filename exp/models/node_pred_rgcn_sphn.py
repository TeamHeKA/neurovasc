import os
import time
import math
import joblib
import numpy as np
import pandas as pd
from statistics import mean, stdev

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, ConfusionMatrixDisplay

import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear

from torch_geometric.data import Data
from torch_geometric.logging import log
from torch_geometric.nn import RGCNConv

def run_rgcn(num_patients, folds, timeOpt, dr, lr, wd, embed_dim, hidden_dim):
    
    entity = pd.read_csv(f'processed_data/sphn_pc_{timeOpt}_entities_{num_patients}.tsv', sep='\t', header=None)
    entity = entity.set_index(entity[1])
    entity = entity.to_dict()[0]

    patients = []
    for i in range(num_patients):
        patient = f'<http://nvasc.org/synth_patient_{i}>'
        patients.append(entity[patient])

    triples = pd.read_csv(f'processed_data/sphn_pc_{timeOpt}_triples_{num_patients}.tsv', sep='\t', header=None)
    triples_inv = triples[[2, 1, 0]]
    triples_inv.columns=[0,1,2]
    triples = triples_inv
    y = np.asarray(joblib.load(f'data/outcomes_pc_{timeOpt}_{num_patients}.joblib'))
    num_x = torch.Tensor(np.load(f'processed_data/sphn_pc_{timeOpt}_numeric_{num_patients}.npy'))

    edge_index = torch.vstack((torch.Tensor(triples[0]).long(),torch.Tensor(triples[2]).long()))
    edge_type = torch.Tensor(triples[1]).long()
    num_nodes = len(entity)
    
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,      
    )
    embedding = Parameter(torch.empty(num_nodes, embed_dim))
    torch.nn.init.xavier_uniform_(embedding, gain=math.sqrt(2.0))
    data.x = embedding
    data.num_x=num_x.view(-1,1)
    data.num_relations = data.num_edge_types
    data.num_classes = 3
    print(data)

    if not os.path.exists(f'results/sphn/exp_rgcn'):
            os.makedirs(f'results/sphn/exp_rgcn')

    metrics = []
    hyper_param = pd.DataFrame(
        {'DROPOUT': [dr], 'LEARNING_RATE': [lr], 'WEIGHT_DECAY': [wd], 'EMBED_DIM': [embed_dim], 'HIDDEN_DIM': [hidden_dim]}
    )
    hyper_param.to_csv(f"results/sphn/exp_rgcn/metrics_pc_{timeOpt}_{num_patients}.csv", mode='a')
    for fold, (train_idx, val_idx, test_idx, train_y, val_y, test_y) in enumerate(zip(*k_fold(np.asarray(patients), y, folds))):

        data.train_idx = torch.Tensor(train_idx).long()
        data.valid_idx = torch.Tensor(val_idx).long()
        data.test_idx = torch.Tensor(test_idx).long()
        
        data.train_y = torch.Tensor(train_y).long()
        data.valid_y = torch.Tensor(val_y).long()
        data.test_y = torch.Tensor(test_y).long()

        # Train RGCN model.
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_lin = Linear(1, embed_dim)
                self.act_lin = torch.nn.PReLU(embed_dim)
                self.conv1 = RGCNConv(embed_dim, hidden_dim, data.num_relations,
                                num_bases=8)
                self.act1 = torch.nn.PReLU(hidden_dim)
                self.conv2 = RGCNConv(hidden_dim, hidden_dim, data.num_relations,
                                num_bases=8)
                self.act2 = torch.nn.PReLU(hidden_dim)
                self.conv3 = RGCNConv(hidden_dim, data.num_classes, data.num_relations,
                                num_bases=8)

            def forward(self, edge_index, edge_type):
                x = self.act_lin(self.num_lin(data.num_x))
                x = x + data.x
                x = self.act1(self.conv1(x, edge_index, edge_type))
                x = F.dropout(x, p=dr, training=self.training)
                x = self.act2(self.conv2(x, edge_index, edge_type))
                x = F.dropout(x, p=dr, training=self.training)
                x = self.conv3(x, edge_index, edge_type)
                return F.log_softmax(x, dim=1)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        model, data = Net().to(device), data.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=wd,
        )

        def train():
            model.train()
            optimizer.zero_grad()
            out = model(data.edge_index, data.edge_type)
            loss = F.nll_loss(out[data.train_idx], data.train_y)
            loss.backward()
            optimizer.step()
            return float(loss)

        @torch.no_grad()
        def test():
            model.eval()
            pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
            train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
            val_acc = float((pred[data.valid_idx] == data.valid_y).float().mean())
            test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
            return train_acc, val_acc, test_acc

        times = []
        best_val_acc = final_test_acc = 0
        for epoch in range(1, 2001):
            start = time.time()
            loss = train()
            train_acc, val_acc, tmp_test_acc = test()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                torch.save(model.state_dict(), f'results/sphn/exp_rgcn/model_weights_pc_{timeOpt}_{num_patients}.pth')
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
            times.append(time.time() - start)
        
        # Evaluation.
        model = Net().to(device)
        model.load_state_dict(torch.load(f'results/sphn/exp_rgcn/model_weights_pc_{timeOpt}_{num_patients}.pth', weights_only=True))
        with torch.no_grad():
            model.eval()
            out = model(data.edge_index, data.edge_type)
            y_pred = out.argmax(dim=-1)
            y_prob = F.softmax(out, dim=-1)
            accuracy = accuracy_score(data.test_y.cpu(), y_pred[data.test_idx].cpu())    
            auc_score_class = roc_auc_score(data.test_y.cpu(), y_prob[data.test_idx].cpu(), average=None, multi_class='ovr')
            auc_score_macro = roc_auc_score(data.test_y.cpu(), y_prob[data.test_idx].cpu(), average='macro', multi_class='ovr')
            auc_score = roc_auc_score(data.test_y.cpu(), y_prob[data.test_idx].cpu(), average='weighted', multi_class='ovr')
            precision_class, recall_class, fscore_class, _ = precision_recall_fscore_support(data.test_y.cpu(), y_pred[data.test_idx].cpu(), average=None)
            precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(data.test_y.cpu(), y_pred[data.test_idx].cpu(), average='macro')
            precision, recall, fscore, _ = precision_recall_fscore_support(data.test_y.cpu(), y_pred[data.test_idx].cpu(), average='weighted')

        metric = pd.DataFrame({
        'PRECISION': np.hstack((precision_class, precision_macro, precision)),
        'RECALL': np.hstack((recall_class, recall_macro, recall)),
        'F1SCORE': np.hstack((fscore_class, fscore_macro, fscore)),
        'ACCURACY': np.hstack((np.zeros(4), accuracy)),
        'AUC': np.hstack((auc_score_class, auc_score_macro, auc_score)),
        }, index=['B2H', 'REHAB', 'DEATH', 'MACRO', 'WEIGHTED'])
        metric.index.name = fold
        metrics.append(metric)
        metric.to_csv(f"results/sphn/exp_rgcn/metrics_pc_{timeOpt}_{num_patients}.csv", mode='a')
        
        if not os.path.exists(f'results/sphn/exp_rgcn/cm'):
            os.makedirs(f'results/sphn/exp_rgcn/cm')

        matrix = confusion_matrix(data.test_y.cpu(), y_pred[data.test_idx].cpu())
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Back2Home", "Reabilitation", "Death"])
        disp.plot()
        fig = disp.figure_
        fig.savefig(f"results/sphn/exp_rgcn/cm/cm_pc_{timeOpt}_{num_patients}_{fold}.jpg", dpi=600)    
    
    panel = pd.concat(metrics)
    metrics_mean = panel.groupby(level=0).mean()
    metrics_mean.index.name = 'MEAN'
    metrics_std = panel.groupby(level=0).std()
    metrics_std.index.name = 'STD'
    metrics_mean.to_csv(f"results/sphn/exp_rgcn/metrics_pc_{timeOpt}_{num_patients}.csv", mode='a')
    metrics_std.to_csv(f"results/sphn/exp_rgcn/metrics_pc_{timeOpt}_{num_patients}.csv", mode='a')


def k_fold(X, y, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=77)
    train_indices, val_indices, test_indices  = [], [], []
    train_y, val_y, test_y = [], [], []
    for (non_test_idx, test_idx) in skf.split(X, y):
        test_indices.append(X[test_idx])
        train_idx, val_idx, _, _ = train_test_split(non_test_idx, y[non_test_idx], test_size=1/9, random_state=77)
        train_indices.append(X[train_idx])
        val_indices.append(X[val_idx])
        train_y.append(y[train_idx])
        val_y.append(y[val_idx])
        test_y.append(y[test_idx])
    return train_indices, val_indices, test_indices, train_y, val_y, test_y