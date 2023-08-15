import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logger import Logger
import os
from trainers import *
from models import LinkPredictor, GNN, Identity, GraphWaveNet
from evaluators import Evaluator
from data_loaders import TrafficAccidentDataset
import time
import itertools

def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    dataset = TrafficAccidentDataset(state_name = args.state_name, data_dir="./data",
                               node_feature_type = args.node_feature_type,
                               use_static_edge_features=args.load_static_edge_features,
                               use_dynamic_node_features=args.load_dynamic_node_features,
                               use_dynamic_edge_features=args.load_dynamic_edge_features,
                               train_years=args.train_years,
                               num_negative_edges=args.num_negative_edges) 
    task_type = "regression" if (args.train_accident_regression or args.train_volume_regression) else "classification"
    evaluator = Evaluator(type=task_type)
    
    data = dataset.data
    in_channels_node = data.x.shape[1] if data.x is not None else 0
    in_channels_node = (in_channels_node + 6) if args.load_dynamic_node_features else in_channels_node
    
    in_channels_edge = data.edge_attr.shape[1] if args.load_static_edge_features else 0
    in_channels_edge = in_channels_edge + 1 if args.load_dynamic_edge_features else in_channels_edge

    if args.encoder == "none":
        model = Identity().to(device)
    if args.encoder == "graph_wavenet":
        model = GraphWaveNet(num_nodes=data.num_nodes, 
                             in_channels_node=in_channels_node, in_channels_edge=in_channels_edge, out_channels=args.hidden_channels, out_timesteps = 1, 
                             dropout=args.dropout, skip_channels=args.hidden_channels, end_channels=args.hidden_channels).to(device)
    else:
        model = GNN(in_channels_node, in_channels_edge, hidden_channels=args.hidden_channels, 
                    num_layers=args.num_gnn_layers, dropout=args.dropout,
                    JK = args.jk_type, gnn_type = args.encoder, num_nodes=data.num_nodes).to(device)
    load_model_dir = os.path.join("./saved/", args.load_model_dir)
    if os.path.exists(os.path.join(load_model_dir, args.load_model_name)):
        model.load_state_dict(
            torch.load(os.path.join(load_model_dir, args.load_model_name), map_location=device)
        )
        print("Loaded model successfully!")
    feature_channels = in_channels_node if args.encoder == "none" else args.hidden_channels
    if_regression = args.train_accident_regression or args.train_volume_regression
    predictor = LinkPredictor(in_channels=feature_channels*2 + in_channels_edge, 
                              hidden_channels=args.hidden_channels, 
                              out_channels=1,
                              num_layers = args.num_predictor_layers,
                              dropout=args.dropout,
                              if_regression=if_regression).to(device)

    results = {}
    for run in range(args.runs):
        predictor.reset_parameters()
        params = itertools.chain(model.parameters(), predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

        if args.train_volume_regression:
            trainer = VolumeRegressionTrainer(model, predictor, dataset, optimizer, evaluator,
                            train_years = args.train_years,
                            valid_years = args.valid_years,
                            test_years = args.test_years,
                            epochs=args.epochs,
                            batch_size = args.batch_size,
                            eval_steps=args.eval_steps,
                            device = device,
                            log_metrics=['MAE', 'MSE'])
        elif args.train_accident_regression:
            trainer = AccidentRegressionTrainer(model, predictor, dataset, optimizer, evaluator,
                            train_years = args.train_years,
                            valid_years = args.valid_years,
                            test_years = args.test_years,
                            epochs=args.epochs,
                            batch_size = args.batch_size,
                            eval_steps=args.eval_steps,
                            device = device,
                            log_metrics=['MAE', 'MSE'], 
                            use_time_series=args.use_time_series, input_time_steps=args.input_time_steps)
        else:
            trainer = Trainer(model, predictor, dataset, optimizer, evaluator,
                            train_years = args.train_years,
                            valid_years = args.valid_years,
                            test_years = args.test_years,
                            epochs=args.epochs,
                            batch_size = args.batch_size,
                            eval_steps=args.eval_steps,
                            device = device,
                            log_metrics=['ROC-AUC', 'F1', 'AP', 'Recall', 'Precision'],
                            use_time_series=args.use_time_series, input_time_steps=args.input_time_steps)
            
        log = trainer.train()

        for key in log.keys():
            if key not in results:
                results[key] = []
            results[key].append(log[key])

    for key in results.keys():
        print("{} : {:.2f} +/- {:.2f}".format(key, np.mean(results[key]), np.std(results[key])))
    
    end = time.time()
    print("Time taken: ", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_accident_regression', action='store_true')
    parser.add_argument('--train_volume_regression', action='store_true')

    parser.add_argument('--state_name', type=str, default="MA")
    parser.add_argument('--num_negative_edges', type=int, default=100000000)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--encoder', type=str, default="none")
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--jk_type', type=str, default="last")
    parser.add_argument('--num_predictor_layers', type=int, default=2)
    parser.add_argument('--input_channels', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--sample_node', action='store_true')
    parser.add_argument('--sample_batch_size', type=int, default=10000)

    parser.add_argument('--batch_size', type=int, default=32*1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--train_years', nargs='+', type=int, default=[2002])
    parser.add_argument('--valid_years', nargs='+', type=int, default=[2003])
    parser.add_argument('--test_years', nargs='+', type=int, default=[2004])

    parser.add_argument('--load_model_dir', type=str, default="none")
    parser.add_argument('--load_model_name', type=str, default="epoch_50.pth")

    # Static node features
    parser.add_argument('--node_feature_type', type=str, default="verse")
    parser.add_argument('--node_feature_name', type=str, default="MA_ppr_128.npy")
    # Other features
    parser.add_argument('--load_static_edge_features', action='store_true')
    parser.add_argument('--load_dynamic_node_features', action='store_true')
    parser.add_argument('--load_dynamic_edge_features', action='store_true')
    # Time Series Arguments
    parser.add_argument('--use_time_series', action='store_true')
    parser.add_argument('--input_time_steps', type=int, default=4)
    args = parser.parse_args()
    print(args)
    main(args)
