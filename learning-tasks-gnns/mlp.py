import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger

from models import LinkPredictor
from evaluators import eval_rocauc
from data_loaders import load_network_with_accidents, load_static_network, load_monthly_data

def train(predictor, x, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(x.device)

        loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size):
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    # Eval ROC-AUC
    train_rocauc = eval_rocauc(pos_train_pred, neg_valid_pred)
    valid_rocauc = eval_rocauc(pos_valid_pred, neg_valid_pred)
    test_rocauc = eval_rocauc(pos_test_pred, neg_test_pred)
    results['ROC-AUC'] = (train_rocauc['ROC-AUC'], valid_rocauc['ROC-AUC'], test_rocauc['ROC-AUC'])
    results['F1'] = (train_rocauc["F1"], valid_rocauc["F1"], test_rocauc["F1"])
    results['AP'] = (train_rocauc["AP"], valid_rocauc["AP"], test_rocauc["AP"])
    
    for K in [1, 3, 5, 10]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


'''
TODO:
- load static features
- normalize weather features
'''

def train_on_month_data(data, predictor, optimizer, batch_size, year, month, device):
    pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = load_monthly_data(data, data_dir="./data", state_name="MA", year=year, month = month)

    new_data = data.clone()
    if new_data.x is None:
        new_data.x = node_features
    else:
        new_data.x = torch.cat([new_data.x, node_features], dim=1)
    
    predictor.train()
    x = new_data.x.to(device)
    pos_train_edge = pos_edges.to(device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        edge = pos_train_edge[perm].t()
        pos_out = predictor(x[edge[0]], x[edge[1]])

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(device)

        print(pos_out, neg_out)
        loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
        loss.backward()
        optimizer.step()
        
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    return total_loss, total_examples

def train(data, predictor, optimizer, batch_size, device, years):
    total_loss = total_examples = 0
    for year in years:
        for month in range(1, 13):
            loss, samples = train_on_month_data(data, predictor, optimizer, batch_size, year, month, device)
            total_loss += loss
            total_examples += samples
    return total_loss/total_examples

@torch.no_grad()
def test_on_month_data(data, predictor, evaluator, batch_size, year, month, device):
    pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = load_monthly_data(data, data_dir="./data", state_name="MA", year=year, month = month)

    new_data = data.clone()
    if new_data.x is None:
        new_data.x = node_features
    else:
        new_data.x = torch.cat([new_data.x, node_features], dim=1)
    
    predictor.eval()

    x = new_data.x.to(device)
    pos_edge = pos_edges.to(device)
    neg_edge = neg_edges.to(device)

    pos_preds = []
    for perm in DataLoader(range(pos_edge.size(0)), batch_size):
        edge = pos_edge[perm].t()
        pos_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_preds = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for perm in DataLoader(range(neg_edge.size(0)), batch_size):
        edge = neg_edge[perm].t()
        neg_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_preds = torch.cat(neg_preds, dim=0)

    results = {}
    # Eval ROC-AUC
    rocauc = eval_rocauc(pos_preds, neg_preds)
    results.update(rocauc)
    
    for K in [1, 3, 5, 10]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_preds,
            'y_pred_neg': neg_preds,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = train_hits

    return results, pos_edges.size(0)

def test(data, predictor, evaluator, batch_size, train_years, valid_years, test_years, device):
    train_results = {}; train_size = 0
    for year in train_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, evaluator, batch_size, year, month, device)
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
            month_results, month_sample_size = test_on_month_data(data, predictor, evaluator, batch_size, year, month, device)
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
            month_results, month_sample_size = test_on_month_data(data, predictor, evaluator, batch_size, year, month, device)
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


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (MLP)')
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
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = PygLinkPropPredDataset(name='ogbl-collab')
    # split_edge = dataset.get_edge_split()
    # data = dataset[0]
    
    # data, split_edge = load_network_with_accidents(data_dir="./data", state_name="MA", 
    #                                             #    train_years=[2002], train_months=[1, 2, 3, 4],
    #                                             #    valid_years=[2002], valid_months=[5, 6, 7, 8],
    #                                             #    test_years=[2002], test_months=[9, 10, 11, 12],
    #                                             num_negative_edges = 1000000,
    #                                             feature_type="verse", feature_name = "MA_ppr_128.npy")
    # print("Number of nodes: {} Number of edges: {}".format(data.num_nodes, data.num_edges))
    # print("Training edges {} Validation edges {} Test edges {}".format(split_edge['train']['edge'].shape[0],
    #                                                                    split_edge['valid']['edge'].shape[0],
    #                                                                    split_edge['test']['edge'].shape[0]))
    # x = data.x
    # x = x.to(device)
    
    data = load_static_network(data_dir="./data", state_name="MA", feature_type="verse", feature_name = "MA_ppr_128.npy")

    predictor = LinkPredictor(args.input_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@1': Logger(args.runs, args),
        'Hits@3': Logger(args.runs, args),
        'Hits@5': Logger(args.runs, args),
        'Hits@10': Logger(args.runs, args),
        'ROC-AUC': Logger(args.runs, args),
        'F1': Logger(args.runs, args),
        'AP': Logger(args.runs, args),
    }

    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            # loss = train(predictor, x, split_edge, optimizer, args.batch_size)
            loss = train(data, predictor, optimizer, args.batch_size, device, years=[2002])

            if epoch % args.eval_steps == 0:
                # results = test(predictor, x, split_edge, evaluator,
                #                args.batch_size)
                results = test(data, predictor, evaluator, args.batch_size, train_years=[2002], valid_years=[2003], test_years=[2004], device=device)
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


if __name__ == "__main__":
    main()
