import inspect

import optuna
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.nn import to_hetero
import torch_geometric.transforms as T

from baselines.util.utils import stratified_cv_split_graph
from model.encoder import WRGCN
from model.regressor import Regressor
from model.shgr import SHGR
from model.transforms import DropEdges, DropFeatures, get_graph_drop_transform
from model.utils import *


class Trainer:
    def __init__(
            self,
            opt,
            model,
            graph
    ):

        self.opt = opt
        self.patience = opt.patience

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.graph = graph

        self.test_data_loader = NeighborLoader(self.graph, num_neighbors=[0], shuffle=False, batch_size=8192)

        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.l2)

        self.transform1 = get_graph_drop_transform(drop_edge_p=opt.de_1, drop_feat_p=opt.df_1)
        self.transform2 = get_graph_drop_transform(drop_edge_p=opt.de_2, drop_feat_p=opt.df_2)

        self.cnt = 0
        self.best_val = -np.inf
        self.best_results = ""

    def __str__(self) -> str:
        return f"Trainer: {self.transform1}, {self.transform2}"

    def reset_metrics(self):
        self.best_val = -np.inf
        self.best_results = {}
        self.cnt = 0

    def evaluate(self, export_embs=False):
        self.model.eval()
        test_mask = self.graph.test_mask

        with torch.no_grad():
            node_representations = self.compute_representations()
            logits = self.model.regressor(node_representations[test_mask]).detach().cpu().numpy()

        targets = self.graph.y[test_mask].cpu()

        mse = mean_squared_error(targets, logits)
        rmse = mean_squared_error(targets, logits, squared=False)
        mae = mean_absolute_error(targets, logits)
        r2 = r2_score(targets, logits)

        results = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

        rmse_per_target = mean_squared_error(targets, logits, squared=False, multioutput='raw_values')
        target_names = ['wond', 'tran', 'tend', 'nost', 'peac', 'joya', 'ener', 'sadn', 'tens']
        for r, name in zip(rmse_per_target, target_names):
            results[f"rmse_{name}"] = r

        if r2 > self.best_val:
            self.best_val = r2
            self.best_results = results
            self.cnt = 0
            if export_embs:
                np.save("embs.npy", node_representations.detach().cpu().numpy())
        else:
            self.cnt += 1

        return results

    def compute_representations(self):
        reps = []

        with torch.no_grad():
            for batch in self.test_data_loader:
                batch = trans_to_cuda(batch)
                rep = self.model.encoder(batch.x, batch.edge_index, batch.edge_type, batch.edge_weight)
                reps.append(rep)

        reps = torch.cat(reps, dim=0)
        return reps

    def train(self, train_loader, epochs, verbose=True, export_embs=False):
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, total_sup_loss, total_unsup_loss, total_consist_loss = 0, 0, 0, 0
            for batch in train_loader:
                batch = trans_to_cuda(batch)
                bs = batch.batch_size

                batch_mask = (torch.sum(batch.y, dim=1).bool() & batch.train_mask)

                # forward
                self.optimizer.zero_grad()

                aug_batch1 = batch.clone()
                aug_batch2 = batch.clone()
                anchor, positive = self.transform1(aug_batch1), self.transform2(aug_batch2)

                anchor_rep = self.model.encoder(anchor.x, anchor.edge_index, anchor.edge_type, anchor.edge_weight)
                pos_rep = self.model.encoder(positive.x, positive.edge_index, positive.edge_type, positive.edge_weight)

                sup_loss, consistency_loss = 0., 0.
                if batch_mask.sum():
                    anchor_support_rep = anchor_rep[:bs][batch_mask[:bs]]
                    pos_support_rep = pos_rep[:bs][batch_mask[:bs]]

                    anchor_rep_filtered = anchor_rep[~batch_mask]
                    pos_rep_filtered = pos_rep[~batch_mask]

                    # consistency loss
                    targets = batch.y[:bs][batch_mask[:bs]]
                    consistency_loss = self.model.consistency_loss(targets, anchor_rep_filtered, pos_rep_filtered,
                                                                   anchor_support_rep, pos_support_rep)

                    # supervised loss
                    rep = (anchor_support_rep + pos_support_rep) / 2
                    logits = self.model.regressor(rep)
                    sup_loss = F.mse_loss(logits, batch.y[:bs][batch_mask[:bs]])

                # unsupervised loss
                unsup_loss = 2 - 2 * F.cosine_similarity(anchor_rep, pos_rep, dim=-1).mean()

                loss = sup_loss + self.opt.alpha * unsup_loss + self.opt.beta * consistency_loss

                loss.backward()
                self.optimizer.step()

                total_loss += float(loss)
                total_sup_loss += float(sup_loss)
                total_unsup_loss += float(self.opt.alpha * unsup_loss)
                total_consist_loss += float(self.opt.beta * consistency_loss)

            # evaluation
            eval_results = self.evaluate(export_embs=export_embs)

            if epoch % 5 == 0 and verbose:
                formatted_results = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in eval_results.items() if not k.startswith("rmse_")])
                st = '[Epoch {}/{}] | Sup_loss : {:.4f} | Unsup_loss : {:.4f} | Cons_loss : {:.4f} | Total_loss : {:.4f} | {}'.format(
                    epoch, epochs, total_sup_loss, total_unsup_loss, total_consist_loss, total_loss, formatted_results)
                print(st)

            if self.cnt == self.patience:
                #print("early stopping!")
                break

        return self.best_results

    def train_folds(self, data, y_labeled, folds=10):
        self.data = data
        self.y_labeled = y_labeled

        num_nodes = data.num_nodes
        stratified_cv = stratified_cv_split_graph(data, folds, seed=self.opt.seed)

        results = []
        for k, (train_index, test_index) in enumerate(stratified_cv):
            # Initialize a new model for each fold
            self.reset_metrics()
            encoder = WRGCN(input_dim=self.opt.input_dim,
                            dim=self.opt.dim,
                            num_layers=self.opt.layers,
                            dropout=self.opt.gnn_dropout,
                            projection=self.opt.projection
                            )
            regressor = Regressor(self.opt.dim if self.opt.layers > 0 or self.opt.projection else self.opt.input_dim,
                                  self.opt.dim,
                                  num_targets=9,
                                  dropout=self.opt.dropout)
            model = trans_to_cuda(SHGR(self.opt, self.y_labeled, encoder=encoder, regressor=regressor))

            self.model = model
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_index] = True
            data.train_mask = train_mask

            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_index] = True
            data.test_mask = test_mask

            train_loader = NeighborLoader(data, num_neighbors=[self.opt.neighbors] * self.opt.layers,
                                          shuffle=True, batch_size=self.opt.batch_size)

            fold_results = self.train(train_loader, epochs=self.opt.epochs, verbose=False)
            results.append(fold_results)
            print(f"FOLD {k + 1} RESULTS: {fold_results}")

        # compute statistics over folds
        mse = np.mean([r["mse"] for r in results])
        rmse = np.mean([r["rmse"] for r in results])
        mae = np.mean([r["mae"] for r in results])
        r2 = np.mean([r["r2"] for r in results])

        # standard deviation
        mse_std = np.std([r["mse"] for r in results])
        rmse_std = np.std([r["rmse"] for r in results])
        mae_std = np.std([r["mae"] for r in results])
        r2_std = np.std([r["r2"] for r in results])

        fold_results = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mse_std": mse_std, "rmse_std": rmse_std,
                        "mae_std": mae_std, "r2_std": r2_std}

        # mean over per target rmse
        target_names = ['wond', 'tran', 'tend', 'nost', 'peac', 'joya', 'ener', 'sadn', 'tens']
        for target in target_names:
            fold_results[f"rmse_{target}"] = np.mean([r[f"rmse_{target}"] for r in results])

        return fold_results
