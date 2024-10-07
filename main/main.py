import argparse
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import numpy as np
import sys
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import random
from utils_HGT import load_dataset, shuffle_data
from model import MiLk-FD, train, test
import time
import os



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='AAAI', help="['AAAI', 'FakeNewsNet', 'ISOT', 'LIAR_PANTS', 'pan2020')")
    
    # GNN related parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.default=200,[50, 100, 150, 200, 300]')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Dim of 1st layer GNN. 32,64,128,256, default=256')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers. 2,3, default=2')
    parser.add_argument('--learning_rate', default=0.0001, help='Learning rate of the optimiser. 0.0001, 0.001, default=0.005')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimiser. 0.001, default=5e-4')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    args = arg_parser()
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hgraph = load_dataset(args.dataset)
    args.device = device
    # print(hgraph.edge_index_dict)
    # start = time.time()

    for i in range(5):
        hgraph = shuffle_data(hgraph, args)
        # print(hgraph.node_types)
        # print(hgraph.metadata())
        # Initialize model parameters
        model = KGT(hgraph, hidden_channels=args.hidden_channels, out_channels=2, num_layers=args.gnn_layers, num_heads=2)
        
        model.to(device)
        hgraph.to(device)
        # kgraph.to(device)
        # Initialize parameters via lazy initialization
        with torch.no_grad():  # Initialize lazy modules.
            out = model(hgraph.x_dict, hgraph.edge_index_dict)
            # out = model(hgraph.x_dict, hgraph.edge_index_dict, kgraph.x_dict, kgraph.edge_index_dict)
            
        train(model, hgraph, args)
        # train(model, hgraph, kgraph, args)

        with torch.no_grad():
            test(model, hgraph, args)
            # test(model, hgraph, kgraph, args)
    # end = time.time()
    # t_time = end - start
    # print(t_time)

        
        
