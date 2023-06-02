import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logger import Logger

from trainers import *
from models import LinkPredictor, GNN, Identity
from evaluators import eval_rocauc, eval_hits
from data_loaders import load_network_with_accidents, load_static_network, load_static_edge_features
import time
import itertools

def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    task_list = []
    task_datas = {}
    task_predictors = {}
    for task_name in args.task_names:
        state_name, data_type, task_type = task_name.split("_")
        data = load_static_network(data_dir="./data", state_name=state_name, 
                                feature_type=args.node_feature_type, 
                                feature_name = f"{state_name}_128.npy" )
        if args.load_static_edge_features:
            data.edge_attr = load_static_edge_features(data_dir="./data", state_name=state_name)

        in_channels_node = data.x.shape[1] if data.x is not None else 0
        in_channels_node = (in_channels_node + 6) if args.load_dynamic_node_features else in_channels_node
        
        in_channels_edge = data.edge_attr.shape[1] if args.load_static_edge_features else 0
        in_channels_edge = in_channels_edge + 1 if args.load_dynamic_edge_features else in_channels_edge
        
        feature_channels = in_channels_node if args.encoder == "none" else args.hidden_channels
        if_regression = task_type == "regression"
        predictor = LinkPredictor(in_channels=feature_channels*2 + in_channels_edge, 
                                hidden_channels=args.hidden_channels, 
                                out_channels=1,
                                num_layers = args.num_predictor_layers,
                                dropout=args.dropout,
                                if_regression=if_regression).to(device)
        
        task_list.append(task_name)
        task_datas[task_name] = data
        task_predictors[task_name] = predictor

    # define encoder
    if args.encoder == "none":
        model = Identity().to(device)
    else:
        # assert in_channels_edge and in_channels_node are same across tasks
        model = GNN(in_channels_node, in_channels_edge, hidden_channels=args.hidden_channels, 
                    num_layers=args.num_gnn_layers, dropout=args.dropout,
                    JK = args.jk_type, gnn_type = args.encoder).to(device)


    # compute mean and std of node & edge features
    node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std = None, None, None, None

    results = {}
    for run in range(args.runs):
        predictor.reset_parameters()
        params = [model.parameters()]
        for task_name in task_list:
            params.append(task_predictors[task_name].parameters())
        params = itertools.chain(*params)
        optimizer = torch.optim.Adam(params, lr=args.lr)

        task_dir = "_".join(task_list)
        checkpoint_dir = f"./saved/{args.encoder}_layer_{args.num_gnn_layers}_dim_{args.hidden_channels}_{task_dir}"
        trainer = MultitaskTrainer(model, optimizer, data_dir="./data",
                        epochs=args.epochs,
                        batch_size = args.batch_size,
                        eval_steps=args.eval_steps,
                        device = device,
                        save_steps=args.save_steps,
                        checkpoint_dir=checkpoint_dir,
                        use_dynamic_node_features=args.load_dynamic_node_features,
                        use_dynamic_edge_features=args.load_dynamic_edge_features,
                        node_feature_mean=node_feature_mean, node_feature_std=node_feature_std,
                        edge_feature_mean=edge_feature_mean, edge_feature_std=edge_feature_std,
                        tasks = task_list, task_to_datas=task_datas, task_to_predictors=task_predictors
                        )
            
        log = trainer.train()
        # node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std = trainer.get_feature_stats()

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

    parser.add_argument('--task_names', nargs='+', default=["MA_accident_classification", "MA_volume_regression"])
    # parser.add_argument('--task_types', nargs='+', default=["accident_classification", "volume_regression"])

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

    parser.add_argument('--save_steps', type=int, default=5)

    # parser.add_argument('--train_years', nargs='+', type=int, default=[2002])
    # parser.add_argument('--valid_years', nargs='+', type=int, default=[2003])
    # parser.add_argument('--test_years', nargs='+', type=int, default=[2004])

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
