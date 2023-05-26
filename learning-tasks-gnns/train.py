import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger

from models import LinkPredictor
from evaluators import eval_rocauc
from data_loaders import load_network_with_accidents, load_static_network, load_monthly_data, load_static_edge_features
import time

'''
TODO:
- load static features
'''

def compute_feature_mean_std(data, data_dir, state_name, years):
    all_node_features = []
    all_edge_features = []
    for year in years:
        for month in range(1, 13):
            _, _, _, node_features, edge_features = load_monthly_data(data, data_dir=data_dir, state_name=state_name, year=year, month = month, num_negative_edges=10000)
            all_node_features.append(node_features)
            all_edge_features.append(edge_features)
        
    all_node_features = torch.cat(all_node_features, dim=0)
    all_edge_features = torch.cat(all_edge_features, dim=0)

    node_feature_mean, node_feature_std = all_node_features.mean(dim=0), all_node_features.std(dim=0)
    edge_feature_mean, edge_feature_std = all_edge_features.mean(dim=0), all_edge_features.std(dim=0)
    return node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std

def train_on_month_data(data, predictor, optimizer, batch_size, year, month, device, 
                        add_node_features = False, add_edge_features = False, 
                        node_feature_mean = None, node_feature_std = None, 
                        edge_feature_mean = None, edge_feature_std = None): 
    pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = load_monthly_data(data, data_dir="./data", state_name="MA", year=year, month = month, num_negative_edges=10000)

    if node_feature_mean is not None:
        node_features = (node_features - node_feature_mean) / node_feature_std
    if edge_feature_mean is not None:
        edge_features = (edge_features - edge_feature_mean) / edge_feature_std

    if pos_edges.size(0) == 0:
        return 0, 0

    new_data = data.clone()
    if add_node_features:
        if new_data.x is None:
            new_data.x = node_features
        else:
            new_data.x = torch.cat([new_data.x, node_features], dim=1)

    if add_edge_features:
        if new_data.edge_attr is None:
            new_data.edge_attr = edge_features
        else:
            new_data.edge_attr = torch.cat([new_data.edge_attr, edge_features], dim=1)
    
    predictor.train()
    x = new_data.x.to(device)
    edge_attr = new_data.edge_attr.to(device) if new_data.edge_attr is not None else None
    pos_train_edge = pos_edges.to(device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        edge = pos_train_edge[perm].t()
        pos_out = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])

        # Sampling from negative edges
        neg_masks = np.random.choice(neg_edges.size(0), edge.size(1), replace=False)
        edge = neg_edges[neg_masks].t() # torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=device)
        neg_out = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(device)

        loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
        loss.backward()
        optimizer.step()
        
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    return total_loss, total_examples

@torch.no_grad()
def test_on_month_data(data, predictor, evaluator, batch_size, year, month, device, 
                        add_node_features = False, add_edge_features = False,
                        node_feature_mean = None, node_feature_std = None,
                        edge_feature_mean = None, edge_feature_std = None):
    pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = load_monthly_data(data, data_dir="./data", state_name="MA", year=year, month = month, num_negative_edges=10000)

    print(f"Eval on {year}-{month} data")
    print(f"Number of positive edges: {pos_edges.size(0)} | Number of negative edges: {neg_edges.size(0)}")

    if node_feature_mean is not None:
        node_features = (node_features - node_feature_mean) / node_feature_std
    if edge_feature_mean is not None:
        edge_features = (edge_features - edge_feature_mean) / edge_feature_std

    if pos_edges.size(0) == 0:
        return {}, 0

    new_data = data.clone()
    if add_node_features:
        if new_data.x is None:
            new_data.x = node_features
        else:
            new_data.x = torch.cat([new_data.x, node_features], dim=1)
    
    if add_edge_features:
        if new_data.edge_attr is None:
            new_data.edge_attr = edge_features
        else:
            new_data.edge_attr = torch.cat([new_data.edge_attr, edge_features], dim=1)
    
    predictor.eval()

    x = new_data.x.to(device)
    edge_attr = new_data.edge_attr.to(device) if new_data.edge_attr is not None else None
    pos_edge = pos_edges.to(device)
    neg_edge = neg_edges.to(device)

    pos_preds = []
    for perm in DataLoader(range(pos_edge.size(0)), batch_size):
        edge = pos_edge[perm].t()
        preds = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])
        pos_preds += [preds.squeeze().cpu()] 
    pos_preds = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for perm in DataLoader(range(neg_edge.size(0)), batch_size):
        edge = neg_edge[perm].t()
        preds = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])
        neg_preds += [preds.squeeze().cpu()]
    neg_preds = torch.cat(neg_preds, dim=0)

    results = {}
    # Eval ROC-AUC
    rocauc = eval_rocauc(pos_preds, neg_preds)
    results.update(rocauc)
    
    for K in [100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_preds,
            'y_pred_neg': neg_preds,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = train_hits

    return results, pos_edges.size(0)

def train(data, predictor, optimizer, batch_size, device, years, 
        add_node_features = False, add_edge_features = False,
        node_feature_mean = None, node_feature_std = None,
        edge_feature_mean = None, edge_feature_std = None):
    total_loss = total_examples = 0
    for year in years:
        for month in range(1, 13):
            loss, samples = train_on_month_data(data, predictor, optimizer, batch_size, year, month, device, 
                                                add_node_features, add_edge_features,
                                                node_feature_mean, node_feature_std,
                                                edge_feature_mean, edge_feature_std)
            total_loss += loss
            total_examples += samples
    return total_loss/total_examples

def test(data, predictor, evaluator, batch_size, train_years, valid_years, test_years, device, 
         add_node_features = False, add_edge_features = False,
         node_feature_mean = None, node_feature_std = None,
         edge_feature_mean = None, edge_feature_std = None):
    train_results = {}; train_size = 0
    for year in train_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, evaluator, batch_size, year, month, device, 
                                                                  add_node_features, add_edge_features,
                                                                  node_feature_mean, node_feature_std,
                                                                  edge_feature_mean, edge_feature_std)
            for key, value in month_results.items():
                if key not in train_results:
                    train_results[key] = 0
                train_results[key] += value * month_sample_size
            train_size += month_sample_size

    for key, value in train_results.items():
        train_results[key] = value / train_size

    val_results = {}; val_size = 0
    for year in valid_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, evaluator, batch_size, year, month, device, 
                                                                  add_node_features, add_edge_features,
                                                                  node_feature_mean, node_feature_std,
                                                                  edge_feature_mean, edge_feature_std)
            for key, value in month_results.items():
                if key not in val_results:
                    val_results[key] = 0
                val_results[key] += value * month_sample_size
            val_size += month_sample_size
    
    for key, value in val_results.items():
        val_results[key] = value / val_size

    test_results = {}; test_size = 0
    for year in test_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, evaluator, batch_size, year, month, device, 
                                                                  add_node_features, add_edge_features,
                                                                  node_feature_mean, node_feature_std,
                                                                  edge_feature_mean, edge_feature_std)
            for key, value in month_results.items():
                if key not in test_results:
                    test_results[key] = 0
                test_results[key] += value * month_sample_size
            test_size += month_sample_size
        
    for key, value in test_results.items():
        test_results[key] = value / test_size

    results = {}
    for key in train_results.keys():
        results[key] = (train_results[key], val_results[key], test_results[key])
    return results


def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    data = load_static_network(data_dir="./data", state_name="MA", 
                               feature_type=args.node_feature_type, 
                               feature_name = args.node_feature_name)
    if args.load_static_edge_features:
        data.edge_attr = load_static_edge_features(data_dir="./data", state_name="MA")
    
    input_channels = data.x.shape[1] if data.x is not None else 0
    input_channels = (input_channels + 6)*2 if args.load_dynamic_node_features else args.input_channels*2
    input_channels = input_channels + data.edge_attr.shape[1] if args.load_static_edge_features else input_channels
    input_channels = input_channels + 1 if args.load_dynamic_edge_features else input_channels
    predictor = LinkPredictor(input_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        # 'Hits@10': Logger(args.runs, args),
        # 'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
        'ROC-AUC': Logger(args.runs, args),
        'F1': Logger(args.runs, args),
        'AP': Logger(args.runs, args),
    }

    # compute mean and std of node & edge features
    if args.load_dynamic_node_features or args.load_dynamic_edge_features:
        node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std = compute_feature_mean_std(data, data_dir="./data", state_name="MA", years = [2002])
    else:
        node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std = None, None, None, None

    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(data, predictor, optimizer, args.batch_size, device, years=[2002], 
                         add_node_features = args.load_dynamic_node_features, add_edge_features = args.load_dynamic_edge_features,
                         node_feature_mean = node_feature_mean, node_feature_std = node_feature_std,
                         edge_feature_mean = edge_feature_mean, edge_feature_std = edge_feature_std)

            if epoch % args.eval_steps == 0:
                results = test(data, predictor, evaluator, args.batch_size, train_years=[2002], valid_years=[2003], test_years=[2004], device=device,
                               add_node_features = args.load_dynamic_node_features, add_edge_features = args.load_dynamic_edge_features,
                               node_feature_mean = node_feature_mean, node_feature_std = node_feature_std,
                               edge_feature_mean = edge_feature_mean, edge_feature_std = edge_feature_std)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
    
    end = time.time()
    print("Time taken: ", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_channels', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=16*1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--node_feature_type', type=str, default="verse")
    parser.add_argument('--node_feature_name', type=str, default="MA_ppr_128.npy")

    parser.add_argument('--load_static_edge_features', action='store_true')
    parser.add_argument('--load_dynamic_node_features', action='store_true')
    parser.add_argument('--load_dynamic_edge_features', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
