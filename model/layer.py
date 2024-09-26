from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param

import torch_geometric.typing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch_geometric.utils.sparse import index2ptr

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class WeightedRGCNConv(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int = 3,
            aggr: str = 'mean',
            root_weight: bool = True,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self.weight = Parameter(
            torch.empty(num_relations, in_channels[0], out_channels))

        if root_weight:
            self.root = Param(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: OptTensor,
                edge_index: Adj, edge_type: OptTensor = None, edge_weight: OptTensor = None):

        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        weight = self.weight

        use_segment_matmul = torch_geometric.backend.use_segment_matmul
        if use_segment_matmul is None:
            segment_count = scatter(torch.ones_like(edge_type), edge_type,
                                    dim_size=self.num_relations)

            self._use_segment_matmul_heuristic_output = (
                torch_geometric.backend.use_segment_matmul_heuristic(
                    num_segments=self.num_relations,
                    max_segment_size=int(segment_count.max()),
                    in_channels=self.weight.size(1),
                    out_channels=self.weight.size(2),
                ))

            assert self._use_segment_matmul_heuristic_output is not None
            use_segment_matmul = self._use_segment_matmul_heuristic_output

        if (use_segment_matmul and torch_geometric.typing.WITH_SEGMM
                    and x_l.is_floating_point()
                    and isinstance(edge_index, Tensor)):
            
            if (edge_type[1:] < edge_type[:-1]).any():
                edge_type, perm = index_sort(
                    edge_type, max_value=self.num_relations)
                edge_index = edge_index[:, perm]

            edge_type_ptr = index2ptr(edge_type, self.num_relations)
            out = self.propagate(edge_index, x=x_l,
                                    edge_type_ptr=edge_type_ptr, edge_weight=edge_weight, size=size)
            
        else:
            for i in range(self.num_relations):
                mask = edge_type == i
                edge_index_i = masked_edge_index(edge_index, mask)
                edge_weight_i = None if edge_weight is None else edge_weight[mask]
                h = self.propagate(edge_index_i, x=x, edge_type_ptr=None, edge_weight=edge_weight_i,
                                size=size)
                out += (h @ weight[i])

        root = self.root
        if root is not None:
            if not torch.is_floating_point(x):
                out = out + root[x]
            else:
                out = out + x @ root

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def message(self, x_j: Tensor, edge_type_ptr: OptTensor, edge_weight: OptTensor = None) -> Tensor:
        if (torch_geometric.typing.WITH_SEGMM and edge_type_ptr is not None):
            if edge_weight is not None:
                return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight) * edge_weight.view(-1, 1)
            
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)
        
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
