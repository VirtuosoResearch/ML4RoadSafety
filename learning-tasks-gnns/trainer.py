import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from evaluators import eval_rocauc, eval_hits
from data_loaders import load_network_with_accidents, load_static_network, load_monthly_data, load_static_edge_features
from logger import Logger

class Trainer:

    def __init__(self, model, predictor, data, optimizer, 
                 data_dir, state_name,
                 train_years, valid_years, test_years,
                 epochs, batch_size, eval_steps, device,
                 log_metrics = ['ROC-AUC', 'F1', 'AP', 'Hits@100'],
                 use_dynamic_node_features = False,
                 use_dynamic_edge_features = False,
                 num_negative_edges = 10000,
                 node_feature_mean = None,
                 node_feature_std = None,
                 edge_feature_mean = None,
                 edge_feature_std = None):
        self.model = model
        self.predictor = predictor
        self.data = data
        self.optimizer = optimizer

        self.data_dir = data_dir
        self.state_name = state_name

        self.train_years = train_years
        self.valid_years = valid_years
        self.test_years = test_years

        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.device = device

        self.use_dynamic_node_features = use_dynamic_node_features
        self.use_dynamic_edge_features = use_dynamic_edge_features
        self.num_negative_edges = num_negative_edges

        # collecting dynamic features normlization statistics
        if node_feature_mean is None and (self.use_dynamic_node_features or self.use_dynamic_edge_features):
            self.node_feature_mean, self.node_feature_std, self.edge_feature_mean, self.edge_feature_std = self.compute_feature_mean_std()
        else:
            self.node_feature_mean = node_feature_mean
            self.node_feature_std = node_feature_std
            self.edge_feature_mean = edge_feature_mean
            self.edge_feature_std = edge_feature_std

        self.loggers = {
            key: Logger(runs=1) for key in log_metrics
        }

    def train_on_month_data(self, year, month): 
        pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(self.data, data_dir=self.data_dir, state_name=self.state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)

        # normalize node and edge features
        if self.node_feature_mean is not None:
            node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        if self.edge_feature_mean is not None:
            edge_features = (edge_features - self.edge_feature_mean) / self.edge_feature_std

        if pos_edges.size(0) == 0:
            return 0, 0

        new_data = self.data.clone()
        if self.use_dynamic_node_features:
            if new_data.x is None:
                new_data.x = node_features
            else:
                new_data.x = torch.cat([new_data.x, node_features], dim=1)

        if self.use_dynamic_edge_features:
            if new_data.edge_attr is None:
                new_data.edge_attr = edge_features
            else:
                new_data.edge_attr = torch.cat([new_data.edge_attr, edge_features], dim=1)
        
        self.model.train()
        self.predictor.train()

        # encoding
        new_data = new_data.to(self.device)
        h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr

        # predicting
        pos_train_edge = pos_edges.to(self.device)
        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
            self.optimizer.zero_grad()
            # positive edges
            edge = pos_train_edge[perm].t()
            pos_out = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])

            # sampling from negative edges
            neg_masks = np.random.choice(neg_edges.size(0), edge.size(1), replace=False)
            edge = neg_edges[neg_masks].t() # torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=device)
            neg_out = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            
            labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(self.device)

            loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
            loss.backward()
            self.optimizer.step()
            
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss, total_examples

    @torch.no_grad()
    def test_on_month_data(self, year, month):
        pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(self.data, data_dir=self.data_dir, state_name=self.state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)

        print(f"Eval on {year}-{month} data")
        print(f"Number of positive edges: {pos_edges.size(0)} | Number of negative edges: {neg_edges.size(0)}")

        # normalize node and edge features
        if self.node_feature_mean is not None:
            node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        if self.edge_feature_mean is not None:
            edge_features = (edge_features - self.edge_feature_mean) / self.edge_feature_std

        if pos_edges.size(0) == 0:
            return {}, 0

        new_data = self.data.clone()
        if self.use_dynamic_node_features:
            if new_data.x is None:
                new_data.x = node_features
            else:
                new_data.x = torch.cat([new_data.x, node_features], dim=1)

        if self.use_dynamic_edge_features:
            if new_data.edge_attr is None:
                new_data.edge_attr = edge_features
            else:
                new_data.edge_attr = torch.cat([new_data.edge_attr, edge_features], dim=1)
        
        self.model.eval()
        self.predictor.eval()

        # encoding
        new_data = new_data.to(self.device)
        h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr

        # predicting
        pos_edge = pos_edges.to(self.device)
        neg_edge = neg_edges.to(self.device)
        pos_preds = []
        for perm in DataLoader(range(pos_edge.size(0)), self.batch_size):
            edge = pos_edge[perm].t()
            preds = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            pos_preds += [preds.squeeze().cpu()] 
        pos_preds = torch.cat(pos_preds, dim=0)

        neg_preds = []
        for perm in DataLoader(range(neg_edge.size(0)), self.batch_size):
            edge = neg_edge[perm].t()
            preds = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            neg_preds += [preds.squeeze().cpu()]
        neg_preds = torch.cat(neg_preds, dim=0)

        results = {}

        # Eval ROC-AUC
        rocauc = eval_rocauc(pos_preds, neg_preds)
        results.update(rocauc)
        # Eval Hits@K
        hits = eval_hits(pos_preds, neg_preds, K=100)
        results.update(hits)

        return results, pos_edges.size(0)


    def train_epoch(self):
        total_loss = total_examples = 0
        for year in self.train_years:
            for month in range(1, 13):
                loss, samples = self.train_on_month_data(year, month)
                total_loss += loss
                total_examples += samples
        return total_loss/total_examples
    

    def train(self):
        train_log = {}
        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch()

            if epoch % self.eval_steps == 0:
                results = self.test()
                for key, result in results.items():
                    self.loggers[key].add_result(run=0, result=result)
            
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_hits:.2f}%, '
                          f'Valid: {100 * valid_hits:.2f}%, '
                          f'Test: {100 * test_hits:.2f}%')
                print('---')

        for key in self.loggers.keys():
            print(key)
            train, valid, test = self.loggers[key].print_statistics(run=0)
            train_log[f"Train_{key}"] = train
            train_log[f"Valid_{key}"] = valid
            train_log[f"Test_{key}"] = test
        return train_log

    def test(self):
        train_results = {}; train_size = 0
        for year in self.train_years:
            for month in range(1, 13):
                month_results, month_sample_size = self.test_on_month_data(year, month)
                for key, value in month_results.items():
                    if key not in train_results:
                        train_results[key] = 0
                    train_results[key] += value * month_sample_size
                train_size += month_sample_size

        for key, value in train_results.items():
            train_results[key] = value / train_size

        val_results = {}; val_size = 0
        for year in self.valid_years:
            for month in range(1, 13):
                month_results, month_sample_size = self.test_on_month_data(year, month)
                for key, value in month_results.items():
                    if key not in val_results:
                        val_results[key] = 0
                    val_results[key] += value * month_sample_size
                val_size += month_sample_size
        
        for key, value in val_results.items():
            val_results[key] = value / val_size

        test_results = {}; test_size = 0
        for year in self.test_years:
            for month in range(1, 13):
                month_results, month_sample_size = self.test_on_month_data(year, month)
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
    
    def compute_feature_mean_std(self):
        all_node_features = []
        all_edge_features = []
        for year in self.train_years:
            for month in range(1, 13):
                _, _, _, node_features, edge_features = \
                    load_monthly_data(self.data, data_dir=self.data_dir, state_name=self.state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)
                all_node_features.append(node_features)
                all_edge_features.append(edge_features)
            
        all_node_features = torch.cat(all_node_features, dim=0)
        all_edge_features = torch.cat(all_edge_features, dim=0)

        node_feature_mean, node_feature_std = all_node_features.mean(dim=0), all_node_features.std(dim=0)
        edge_feature_mean, edge_feature_std = all_edge_features.mean(dim=0), all_edge_features.std(dim=0)
        return node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std
    
    def get_feature_stats(self):
        return self.node_feature_mean, self.node_feature_std, self.edge_feature_mean, self.edge_feature_std