from typing import List, Optional, Tuple, Union

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from histocartography.interpretability.base_explainer import BaseExplainer
from histocartography.interpretability.grad_cam import GradCAM

class BaseGraphGradCAMExplainer(BaseExplainer):
    def __init__(
        self,
        gnn_layer_name: List[str] = None,
        gnn_layer_ids: List[str] = None,
        **kwargs
    ) -> None:
        """
        BaseGraphGradCAMExplainer explainer constructor.

        Args:
            gnn_layer_name (List[str]): List of reference layers to use for computing CAM
                                        Default to None. If None tries to automatically infer
                                        from the model.
            gnn_layer_ids: (List[str]): List of reference layer IDs to use for computing CAM
                                        Default to None. If None tries to automatically infer
                                        from the model.
        """
        super().__init__(**kwargs)
        if gnn_layer_name is None and gnn_layer_ids is None:
            all_param_names = [
                name for name,
                _ in self.model.named_parameters()]
            self.gnn_layer_ids = list(filter(lambda x: x.isdigit(), set(
                [p.split(".")[2] for p in all_param_names])))
            self.gnn_layer_name = all_param_names[0].split(".")[0]
        else:
            self.gnn_layer_ids = gnn_layer_ids
            self.gnn_layer_name = gnn_layer_name

        assert self.gnn_layer_ids is not None
        assert self.gnn_layer_name is not None

    def _process(
        self, graph: dgl.DGLGraph, class_idx: Union[None, int, List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node importances for a single class

        Args:
            graph (dgl.DGLGraph): Graph to explain.
            class_idx (Union[None, int, List[int]]): Class indices (index) to explain. If None results in using the winning class.
                                                     If a list is provided, explainer all the class indices provided.
                                                     Defaults to None.

        Returns:
            node_importance (np.ndarray): Node-level importance scores.
            logits (np.ndarray): Prediction logits.
        """
        if isinstance(class_idx, int) or class_idx is None:
            class_idx = [class_idx]
        node_importances, logits = self._process_all(graph, class_idx)
        return node_importances, logits

    def _get_extractor(self):
        raise NotImplementedError("Abstract class.")

    def _process_all(
        self, graph: dgl.DGLGraph, classes: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node importances for all classes

        Args:
            graph (dgl.DGLGraph): Graph to explain
            classes (List[int]): Classes to explain

        Returns:
            node_importance (np.ndarray): Node-level importance scores
            logits (np.ndarray): Prediction logits
        """
        self.extractor = self._get_extractor()
        original_logits = self.model(graph)
        if isinstance(original_logits, tuple):
            original_logits = original_logits[0]
        if classes[0] is None:
            classes = [original_logits.argmax().item()]
        all_class_importances = list()
        for class_idx in classes:
            node_importance = self.extractor(
                class_idx, original_logits, normalized=True
            ).cpu()
            all_class_importances.append(node_importance)
            self.extractor.clear_hooks()
        logits = original_logits.cpu().detach().numpy()
        node_importances = torch.stack(all_class_importances)
        node_importances = node_importances.cpu().detach().squeeze(dim=0).numpy()
        return node_importances, logits


class GraphGradCAMExplainer(BaseGraphGradCAMExplainer):
    """
    Explain a graph with GradCAM.
    """

    def _get_extractor(self):
        return GradCAM(
            getattr(self.model, self.gnn_layer_name).layers, self.gnn_layer_ids
        )
