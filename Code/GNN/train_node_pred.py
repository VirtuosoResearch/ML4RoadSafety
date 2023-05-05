import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.datasets import Coauthor, Flickr, Reddit2, Planetoid, CoraFull
from torch_geometric.utils import to_scipy_sparse_matrix, degree, to_undirected
from torch_geometric.data import Data
import time

from model import GCN, SAGE, GAT, GIN
from utils.pagerank import pagerank_scipy
from utils.util import set_train_val_test_split
from trainer import Trainer, DROTrainer, TAWTrainer, SAMTrainer, SupConTrainer, LabelSmoothTrainer, ConstraintTrainer, EnsembleTrainer
from utils.sam import SAM
from utils.sample import effective_resistance_sample_graph, graphsaint_sample_graph

def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = Planetoid(path, args.dataset, transform=T.Compose([T.NormalizeFeatures()]))

        data = dataset[0]
        data.y = data.y.unsqueeze(1)
        train_idx = torch.nonzero(data.train_mask).squeeze().to(device)
        valid_idx = torch.nonzero(data.val_mask).squeeze().to(device)
        test_idx = torch.nonzero(data.test_mask).squeeze().to(device)
        split_idx = {'train':train_idx, 'valid':valid_idx, 'test':test_idx}
    elif args.dataset in ['arxiv', 'products', 'proteins', 'mag']:
        name = 'ogbn-' + args.dataset
        dataset = PygNodePropPredDataset(name=name, transform=T.ToUndirected())
        data = dataset[0]

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train'].to(device)
    elif args.dataset == 'coauthor':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'coauthor')
        dataset = Coauthor(path, 'CS', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures()]))
        data = dataset[0]

        data = set_train_val_test_split(2406525885, data, 5000, 20)
        data.y = data.y.unsqueeze(1)
        train_idx = torch.nonzero(data.train_mask).squeeze().to(device)
        valid_idx = torch.nonzero(data.val_mask).squeeze().to(device)
        test_idx = torch.nonzero(data.test_mask).squeeze().to(device)
        split_idx = {'train':train_idx, 'valid':valid_idx, 'test':test_idx}
    elif args.dataset == 'flickr':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'flickr')
        dataset = Flickr(path, transform=T.Compose([T.ToUndirected()]))
        data = dataset[0]

        data.y = data.y.unsqueeze(1)
        train_idx = torch.nonzero(data.train_mask).squeeze().to(device)
        valid_idx = torch.nonzero(data.val_mask).squeeze().to(device)
        test_idx = torch.nonzero(data.test_mask).squeeze().to(device)
        split_idx = {'train':train_idx, 'valid':valid_idx, 'test':test_idx}
    elif args.dataset == 'reddit':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'reddit')
        dataset = Reddit2(path, transform=T.Compose([T.ToUndirected()]))
        data = dataset[0]

        data.y = data.y.unsqueeze(1)
        train_idx = torch.nonzero(data.train_mask).squeeze().to(device)
        valid_idx = torch.nonzero(data.val_mask).squeeze().to(device)
        test_idx = torch.nonzero(data.test_mask).squeeze().to(device)
        split_idx = {'train':train_idx, 'valid':valid_idx, 'test':test_idx}
    else:
        print("Non-valid dataset name!")
        exit()
    
    # Compute the degrees
    transform = T.ToSparseTensor(remove_edge_index=False)
    data = transform(data)
    data.adj_t = data.adj_t.to_symmetric()
    degrees = data.adj_t.sum(0)
    degree_thres = torch.quantile(degrees, 0.25) # threshold of lower 25% of the nodes degree

    # Sampling
    if args.ensemble:
        iters = args.num_ensemble
    else:
        iters = 1
    
    datas = []
    original_data = data.clone()
    for _ in range(iters):
        if args.sample_method == 'effective_resistance':
            data = original_data.clone()
            data.edge_index, data.edge_weight = effective_resistance_sample_graph(original_data, args.alpha, args.approx_score, int(np.ceil(args.k*np.log(100*original_data.num_nodes))))
            data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
            data = transform(data)
            data.adj_t = data.adj_t.to_symmetric()
        elif args.sample_method == 'graphsaint':
            data = original_data.clone()
            data.edge_index, data.edge_weight = graphsaint_sample_graph(original_data, int(args.sample_ratio*original_data.num_edges))
            data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, num_nodes=data.num_nodes)
            data = transform(data)
            data.adj_t = data.adj_t.to_symmetric()
        else:
            data.edge_weight = None
        
        datas.append(data)

    # Initialize the model
    if args.model == "sage":
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.model == 'gat':
        model = GAT(data.num_features, int(args.hidden_channels/args.num_heads),
                    dataset.num_classes, args.num_layers,
                    args.num_heads, args.dropout,
                    args.input_drop, args.attn_drop).to(device)
    elif args.model == 'gin':
        model = GIN(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, use_bn=not args.no_bn).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')

    # if args.reweight_degree:
    #     degrees = data.adj_t.sum(0)
    #     weights = torch.ones_like(degrees)
    #     weights[degrees<=args.degree_thres] = args.upweight
    #     # weights = 1/(torch.log2(degrees+1))
    # elif args.reweight_pagerank:
    #     ''' Compute pagerank '''
    #     G = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    #     pageranks = pagerank_scipy(G, alpha=args.pagerank_alpha)
    #     pageranks = torch.Tensor(pageranks).to(device)
    #     weights = 1/pageranks
    # else:
    #     weights = torch.ones(data.x.size(0))
    # weights = weights.to(device)

    test_accs = []; longtail_accs = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam([
            dict(params=model.convs[0].parameters(), weight_decay=args.weight_decay),
            dict(params=model.convs[1:].parameters(), weight_decay=0)
        ], lr=args.lr)
        
        if args.train_constraint:
            trainer = ConstraintTrainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degree_thres=args.degree_thres, monitor=args.monitor,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_constraint")
            trainer.add_constraint(
                lambda_extractor = args.reg_weight, norm='frob', state_dict = None
            )
        elif args.train_sam:
            optimizer = SAM(model.parameters(), torch.optim.Adam, rho=args.sam_rho, adaptive=False, lr=args.lr)
            trainer = SAMTrainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degree_thres=args.degree_thres, monitor=args.monitor,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_sam")
        elif args.train_supcon:
            trainer = SupConTrainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degree_thres=args.degree_thres, monitor=args.monitor,
                            supcon_lam = args.supcon_lam, supcon_tmp = args.supcon_tmp,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_supcon")
        elif args.train_ls:
            trainer = LabelSmoothTrainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degree_thres=args.degree_thres, monitor=args.monitor,
                            num_classes=dataset.num_classes, alpha=args.ls_alpha,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_ls")
        elif args.train_dro:
            trainer = DROTrainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degree_thres=args.degree_thres, monitor=args.monitor,
                            group_by=args.group_by, group_num=args.group_num, weight_lr=args.weight_lr, group_adj=args.group_adj,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_dro_by_{args.group_by}") # _run_{run}
        elif args.train_tawt:
            trainer = TAWTrainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degree_thres=args.degree_thres, monitor=args.monitor,
                            group_by=args.group_by, group_num=args.group_num, weight_lr=args.weight_lr,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_tawt_by_{args.group_by}")
        elif args.ensemble:
            trainer = EnsembleTrainer(model, optimizer, datas, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degrees=degrees, degree_thres=degree_thres, monitor=args.monitor,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_ensemble")
        else:
            trainer = Trainer(model, optimizer, data, split_idx, evaluator, device,
                            epochs=args.epochs, log_steps=args.log_steps, degrees=degrees ,degree_thres=degree_thres, monitor=args.monitor,
                            checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_run_{run}") # _run_{run}
        test_acc, longtail_acc = trainer.train()
        test_accs.append(test_acc)
        longtail_accs.append(longtail_acc)
    print("Test accuracy: {:.4f}±{:.4f}".format(np.mean(test_accs), np.std(test_accs)))
    print("Test accuracy for degree <={:2.0f}: {:.4f}±{:.4f}".format(
            degree_thres, np.mean(longtail_accs), np.std(longtail_accs)
        ))
    print(f"Total time: {time.time()-start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--degree_thres', type=int, default=6)
    parser.add_argument('--degree_power', type=int, default=10)
    parser.add_argument('--use_edge_index', action="store_true")
    parser.add_argument('--monitor', type=str, default="accuracy")

    ''' Sampling '''
    parser.add_argument('--sample_method', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--approx_score', type=bool, default=True)
    parser.add_argument('--k', type=int, default=24)
    parser.add_argument('--sample_ratio', type=float, default=1)

    ''' Model '''
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no_bn', action="store_true")

    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--input_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', default=0.1)

    ''' Training '''
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0)
    
    ''' Ensemble '''
    parser.add_argument('--ensemble', action="store_true")
    parser.add_argument('--num_ensemble', type=int, default=3)

    ''' Reweighting '''
    parser.add_argument('--reweight_degree', action="store_true")
    parser.add_argument('--reweight_pagerank', action="store_true")
    parser.add_argument('--pagerank_alpha', type=float, default=0.15)
    parser.add_argument('--upweight', type=float, default=1.5)

    parser.add_argument("--train_dro", action="store_true")
    parser.add_argument("--group_by", type=str, default="degree")
    parser.add_argument("--group_num", type=int, default=4)
    parser.add_argument("--weight_lr", type=float, default=1)
    parser.add_argument("--group_adj", type=float, default=0)

    parser.add_argument('--train_tawt', action="store_true")

    ''' Regularization '''
    parser.add_argument('--train_ls', action="store_true")
    parser.add_argument('--ls_alpha', type=float, default=0.15)

    parser.add_argument('--train_constraint', action="store_true")
    parser.add_argument('--reg_weight', type=float, default=0.1)

    parser.add_argument('--train_sam', action="store_true")
    parser.add_argument('--sam_rho', type=float, default=0.05)

    parser.add_argument('--train_supcon', action="store_true")
    parser.add_argument('--supcon_lam', type=float, default=0.9)
    parser.add_argument('--supcon_tmp', type=float, default=0.3)

    args = parser.parse_args()
    print(args)
    main(args)
