import argparse

from requests import get
from sklearn.model_selection import train_test_split
import torch

import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from params import get_hyperparameters
from preprocessing.utils import *
from model.utils import *
import pickle
import csv

from model.shgr import SHGR
from model.trainer import Trainer
from model.encoder import *
from model.regressor import Regressor

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default='sample', help='sample/music4all-onion/m4a')
parser.add_argument('--rep', type=str, default="musicnn", help='musicnn/jukebox/maest')

parser.add_argument('--input-dim', type=int, default=50)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--layers', type=int, default=1, help='the number of gnn layers')
parser.add_argument('--gnn-dropout', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--projection', type=int, default=0)

parser.add_argument('--neighbors', type=int, default=20)

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')

parser.add_argument("--df_1", type=float, default=0.0)
parser.add_argument("--de_1", type=float, default=0.0)
parser.add_argument("--df_2", type=float, default=0.0)
parser.add_argument("--de_2", type=float, default=0.0)

parser.add_argument('--clusters', type=int, default=8)
parser.add_argument('--k', '-k', type=int, default=8)
parser.add_argument('--confidence-threshold', '-ct', type=float, default=0.1)
parser.add_argument('--tau', '-t', type=float, default=0.1)

parser.add_argument('--alpha', '-a', type=float, default=0.0)
parser.add_argument('--beta', '-b', type=float, default=0.0)

parser.add_argument("--epochs", '-e', type=int, default=1000)
parser.add_argument("--batch-size", '-bs', type=int, default=100)
parser.add_argument('--patience', type=int, default=30)

parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--folds', type=int, default=10)

opt = parser.parse_args()

def main(opt):
    opt = get_hyperparameters(opt, dataset=opt.dataset, rep=opt.rep)

    init_seed(opt.seed)
    print(opt)

    features, label_dict = load_preprocessed_data('data/' + opt.dataset, feature_file=f"features-{opt.rep}.npy")
    y_node_ids = np.array(list(label_dict.keys()))
    y_labeled = np.array(list(label_dict.values()))

    y = torch.zeros((opt.num_node, 9))
    y[y_node_ids] = torch.FloatTensor(y_labeled)

    train_indexes, test_indexes = train_test_split(y_node_ids, test_size=opt.ratio, random_state=opt.seed)

    train_mask = torch.zeros(opt.num_node, dtype=torch.bool)
    test_mask = torch.zeros(opt.num_node, dtype=torch.bool)
    train_mask[train_indexes] = True
    test_mask[test_indexes] = True

    edge_index, edge_weight, edge_type = pickle.load(open('data/' + opt.dataset + '/graph.pkl', 'rb'))
    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.FloatTensor(edge_weight)
    edge_type = torch.LongTensor(edge_type)

    print("before data")

    data = Data(
        x=torch.FloatTensor(features),
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        train_mask=train_mask,
        test_mask=test_mask,
        label_mask=train_mask | test_mask
    )
    print(data)

    train_loader = NeighborLoader(data, num_neighbors=[opt.neighbors] * opt.layers,
                                  shuffle=True, batch_size=opt.batch_size)

    print("after loader")
    encoder = WRGCN(input_dim=opt.input_dim,
                    dim=opt.dim,
                    num_layers=opt.layers,
                    dropout=opt.gnn_dropout,
                    projection=opt.projection
                    )

    regressor = Regressor(opt.dim if opt.layers > 0 or opt.projection else opt.input_dim, opt.dim, num_targets=9,
                          dropout=opt.dropout)
    
    print("before model")

    model = trans_to_cuda(SHGR(opt, y_labeled, encoder=encoder, regressor=regressor))
    print(model)

    trainer = Trainer(
        opt,
        model,
        data,
    )

    print('start training')

    results = trainer.train(train_loader, opt.epochs, export_embs=False)
    #results = trainer.train_folds(data, y_labeled, folds=opt.folds)

    for key, value in results.items():
        print(f'{key}: {value:.4f}')

    print('\n')
    print(opt)


if __name__ == '__main__':
    main(opt)
