import argparse

import numpy as np
import random

import torch
import torch.nn.functional as F

from model import GCN
from utils.random_graphs import generate_ba_graph
from utils.util import test_longtail_performance

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.shape[0]
        accs.append(acc)
    return accs

def main(args):
    ''' Load dataset'''
    feature_dim = args.feature_dim
    class_ratio = args.class_ratio
    seed = 42; random.seed(seed); np.random.seed(seed) # fix random seed in generating the graphs (for results reproducibility)
    data = generate_ba_graph(
        node_num=args.node_num, edge_density=args.edge_num, 
        feature_dim=feature_dim, class_ratio=class_ratio, train_ratio=args.train_ratio)

    ''' Initialize model '''
    in_channels = feature_dim
    out_channels = 2
    model = GCN(in_channels=in_channels, 
            hidden_channels=args.hidden,
            out_channels=out_channels,
            num_layers=args.num_layers,
            dropout=args.dropout)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    model, data = model.to(device), data.to(device)

    longtail_accs = []
    for run in range(args.runs):
        model.reset_parameters()    
        optimizer = torch.optim.Adam([
            dict(params=model.convs[0].parameters(), weight_decay=args.weight_decay),
            dict(params=model.convs[1:].parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.

        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs):
            train(model, optimizer, data)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
        longtail_accs.append(test_longtail_performance(model, data))
    longtail_accs = np.stack(longtail_accs, axis=0)
    acc_means = np.mean(longtail_accs, axis = 0)
    acc_stds = np.std(longtail_accs, axis = 0)
    for i in range(acc_means.shape[0]):
        print("\t",
            "accuracy= {:.4f}".format(acc_means[i]),
            "std= {:.4f}".format(acc_stds[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--runs', type=int, default=10)

    ''' Generating data '''
    parser.add_argument('--node_num', type=int, default=10000)
    parser.add_argument('--edge_num', type=int, default=5)
    parser.add_argument('--feature_dim', type=int, default=16)
    parser.add_argument('--class_ratio', type=float, default=0.6)
    parser.add_argument('--train_ratio', type=float, default=0.1)

    ''' Training '''
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    ''' Model '''
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    main(args)