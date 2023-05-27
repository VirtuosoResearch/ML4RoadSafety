import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logger import Logger

from trainer import Trainer
from models import LinkPredictor, GNN, Identity
from evaluators import eval_rocauc, eval_hits
from data_loaders import load_network_with_accidents, load_static_network, load_static_edge_features
import time

'''
TODO:
- load static features
- Write GCN encoder: while also taking the edge features into account
'''

def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    data = load_static_network(data_dir="./data", state_name="MA", 
                               feature_type=args.node_feature_type, 
                               feature_name = args.node_feature_name)
    if args.load_static_edge_features:
        data.edge_attr = load_static_edge_features(data_dir="./data", state_name="MA")
    
    in_channels_node = data.x.shape[1] if data.x is not None else 0
    in_channels_node = (in_channels_node + 6) if args.load_dynamic_node_features else in_channels_node
    
    in_channels_edge = data.edge_attr.shape[1] if args.load_static_edge_features else 0
    in_channels_edge = in_channels_edge + 1 if args.load_dynamic_edge_features else in_channels_edge

    if args.encoder == "none":
        model = Identity().to(device)
    else:
        model = GNN(in_channels_node, in_channels_edge, hidden_channels=args.hidden_channels, 
                    num_layers=args.num_gnn_layers, dropout=args.dropout,
                    JK = args.jk_type, gnn_type = args.encoder).to(device)
    feature_channels = in_channels_node if args.encoder == "none" else args.hidden_channels
    predictor = LinkPredictor(in_channels=feature_channels*2 + in_channels_edge, 
                              hidden_channels=args.hidden_channels, 
                              out_channels=1,
                              num_layers = args.num_predictor_layers,
                              dropout=args.dropout).to(device)

    # compute mean and std of node & edge features
    node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std = None, None, None, None

    results = {}
    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        trainer = Trainer(model, predictor, data, optimizer,
                        data_dir="./data", state_name="MA",
                        train_years = args.train_years,
                        valid_years = args.valid_years,
                        test_years = args.test_years,
                        epochs=args.epochs,
                        batch_size = args.batch_size,
                        eval_steps=args.eval_steps,
                        device = device,
                        use_dynamic_node_features=args.load_dynamic_node_features,
                        use_dynamic_edge_features=args.load_dynamic_edge_features,
                        log_metrics=['ROC-AUC', 'F1', 'AP', 'Hits@100'],
                        node_feature_mean=node_feature_mean, node_feature_std=node_feature_std,
                        edge_feature_mean=edge_feature_mean, edge_feature_std=edge_feature_std,)
        
        log = trainer.train()
        node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std = trainer.get_feature_stats()

        for key in log.keys():
            if key not in results:
                results[key] = []
            results[key].append(log[key])

    for key in results.keys():
        print("{} : {:.2f} +/- {:.2f}".format(key, 100 * np.mean(results[key]), 100 * np.std(results[key])))
    
    end = time.time()
    print("Time taken: ", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--encoder', type=str, default="none")
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--jk_type', type=str, default="last")
    parser.add_argument('--num_predictor_layers', type=int, default=2)
    parser.add_argument('--input_channels', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=16*1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--train_years', nargs='+', type=int, default=[2002])
    parser.add_argument('--valid_years', nargs='+', type=int, default=[2003])
    parser.add_argument('--test_years', nargs='+', type=int, default=[2004])

    # Static node features
    parser.add_argument('--node_feature_type', type=str, default="verse")
    parser.add_argument('--node_feature_name', type=str, default="MA_ppr_128.npy")
    # Other features
    parser.add_argument('--load_static_edge_features', action='store_true')
    parser.add_argument('--load_dynamic_node_features', action='store_true')
    parser.add_argument('--load_dynamic_edge_features', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
