import os
import torch
import argparse
import numpy as np
from glob import glob
import json
import yaml
from tqdm import tqdm
from dgl.data.utils import load_graphs
from histocartography.interpretability import GraphLRPExplainer as GraphGradCAMExplainer
from histocartography.utils.graph import set_graph_on_cuda
from cell_graph_model import CustomCellGraphModel

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        required=True
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        help='path to where the model is saved.',
        required=True
    )
    parser.add_argument(
        '--output',
        type=str,
        help='path to where the result should be saved.',
        required=True
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='the cuda device to be used '
    )

    return parser.parse_args()


def main(args, config):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = CustomCellGraphModel(
        gnn_params=config['gnn_params'],
        classification_params=config['classification_params'],
        node_dim=NODE_DIM,
        num_classes=7
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Define the explainer
    explainer = GraphGradCAMExplainer(model=model)

    # Load preprocessed data
    cg_fnames = glob(os.path.join(args.cg_path, '*.bin'))

    cg_fnames.sort()

    output = []
    for cg_name in tqdm(cg_fnames, desc='Explaining the prediction using GraphCAM', unit='graph'):

        graph, _ = load_graphs(cg_name)
        graph = graph[0]
        graph = set_graph_on_cuda(graph) if torch.cuda.is_available() else graph

        importance_scores, logits = explainer.process(graph, output_name=cg_name.replace('.bin', ''))
        print('logits: ', logits)

        output.append({
            'graph': graph,
            'importance_scores': importance_scores,
            'logits': logits
        })

    with open(args.output, 'w') as json_file:
        json.dump(output, json_file)

if __name__ == '__main__':
    args = parse_arguments()

    with open(args.config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.device:
        DEVICE=args.device

    main(args, config)
