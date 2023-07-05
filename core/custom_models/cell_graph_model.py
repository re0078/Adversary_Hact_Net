""" 
Adopted from histocartography.ml
"""
import dgl
import os
import torch
from typing import Tuple, Union, List

from histocartography.ml import CellGraphModel
from .adversarial import projected_gradient_descent

GNN_NODE_FEAT_IN = 'feat'

def show_tensor_stats(tensor):
    print("Tensor Shape:", tensor.shape)
    print("Tensor Size:", tensor.size())
    print("Tensor Dimension:", tensor.dim())
    print("Tensor Maximum:", tensor.max().item())
    print("Tensor Minimum:", tensor.min().item())
    print("Tensor Mean:", tensor.mean().item())
    print("Tensor Standard Deviation:", tensor.std().item())
    print("Tensor Sum:", tensor.sum().item())

class CustomCellGraphModel(CellGraphModel):
    def __init__(self, epsilon:float=0.01, **kwargs):
        CellGraphModel.__init__(self, **kwargs)
        self.epsilon = epsilon

    def forward(
        self,
        graph: Union[dgl.DGLGraph,
                     Tuple[torch.tensor, torch.tensor]],
        adversarial: bool=False,
        labels: torch.Tensor=None,
        loss_fn: torch.nn.Module=None
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            graph (Union[dgl.DGLGraph, Tuple[torch.tensor, torch.tensor]]): Cell graph to process.

        Returns:
            torch.tensor: Model output.
        """

        # 1. GNN layers over the cell graph
        if isinstance(graph, dgl.DGLGraph):
            feats = graph.ndata[GNN_NODE_FEAT_IN]
            graph_embeddings = self.cell_graph_gnn(graph, feats)
        else:
            adj, feats = graph[0], graph[1]
            graph_embeddings = self.cell_graph_gnn(adj, feats)

        if adversarial:
            #show_tensor_stats(graph_embeddings)
            graph_embeddings = projected_gradient_descent(self, graph_embeddings, labels, loss_fn, 
                                num_steps=40, step_size=0.1,
                                eps=self.epsilon, eps_norm=2,
                                step_norm=2)

        # 2. Run readout function
        out = self.pred_layer(graph_embeddings)

        return out