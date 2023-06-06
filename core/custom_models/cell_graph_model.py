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


class CustomCellGraphModel(CellGraphModel):
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
            graph_embeddings = projected_gradient_descent(self, graph_embeddings, labels, loss_fn, 
                                num_steps=40, step_size=0.01,
                                eps=5e-05, eps_norm='inf',
                                step_norm='inf')

        # 2. Run readout function
        out = self.pred_layer(graph_embeddings)

        return out