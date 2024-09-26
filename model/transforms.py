import copy
import torch
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.transforms import Compose
from torch_geometric.data import HeteroData


class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        if isinstance(data, HeteroData):
            node_types, edge_types = data.metadata()
            for node_type in node_types:
                if node_type == 'n_id':
                    continue
                drop_mask = torch.empty((data[node_type].x.size(1),), dtype=torch.float32,
                                        device=data[node_type].x.device).uniform_(0, 1) < self.p
                data[node_type].x[:, drop_mask] = 0
        else:
            drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
            data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        if isinstance(data, HeteroData):
            node_types, edge_types = data.metadata()
            for edge_type in edge_types:
                edge_index = data.edge_index_dict[edge_type]
                edge_weight = data.edge_weight_dict[edge_type] if len(data.edge_weight_dict) > 0 else None

                edge_index, edge_id = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

                data[edge_type].edge_index = edge_index
                if edge_weight is not None:
                    data[edge_type].edge_weight = edge_weight[edge_id]
        else:
            edge_index = data.edge_index
            edge_attr = data.edge_attr if 'edge_attr' in data else None
            edge_weight = data.edge_weight if 'edge_weight' in data else None
            edge_type = data.edge_type if 'edge_type' in data else None

            edge_index, edge_id = dropout_edge(edge_index, p=self.p, force_undirected=self.force_undirected)

            data.edge_index = edge_index
            if edge_attr is not None:
                data.edge_attr = edge_attr[edge_id]
            if edge_weight is not None:
                data.edge_weight = edge_weight[edge_id]
            if edge_type is not None:
                data.edge_type = edge_type[edge_id]

        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.copy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)