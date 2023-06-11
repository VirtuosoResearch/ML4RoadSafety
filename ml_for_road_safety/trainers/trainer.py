import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from logger import Logger

class Trainer:

    def __init__(self, model, predictor, dataset, optimizer, evaluator,
                 train_years, valid_years, test_years,
                 epochs, batch_size, eval_steps, device,
                 log_metrics = ['ROC-AUC', 'F1', 'AP', 'Recall', 'Precision']):
        self.model = model
        self.predictor = predictor
        self.dataset = dataset
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.train_years = train_years
        self.valid_years = valid_years
        self.test_years = test_years

        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.device = device

        self.loggers = {
            key: Logger(runs=1) for key in log_metrics
        }

    def train_on_month_data(self, year, month): 
        monthly_data = self.dataset.load_monthly_data(year, month)

        new_data = monthly_data['data']
        pos_edges, pos_edge_weights, neg_edges = \
            monthly_data['accidents'], monthly_data['accident_counts'], monthly_data['neg_edges']
        
        if pos_edges is None or pos_edges.size(0) < 10:
            return 0, 0
        
        self.model.train()
        self.predictor.train()

        # encoding
        new_data = new_data.to(self.device)
        edge_attr = new_data.edge_attr
        h = self.model(new_data.x, new_data.edge_index, edge_attr)

        # predicting
        pos_train_edge = pos_edges.to(self.device)
        pos_edge_weights = pos_edge_weights.to(self.device)
        neg_edges = neg_edges.to(self.device)
        total_loss = total_examples = 0
        # self.batch_size > pos_train_edge.size(0): only backprop once since it does not retain cache.
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
            self.optimizer.zero_grad()
            # positive edges
            edge = pos_train_edge[perm].t()
            pos_out = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])

            # sampling from negative edges
            neg_masks = np.random.choice(neg_edges.size(0), min(edge.size(1), neg_edges.size(0)), replace=False)
            edge = neg_edges[neg_masks].t() # torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=device)
            neg_out = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            
            labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(self.device)
            loss = self.evaluator.criterion(torch.cat([pos_out, neg_out]), labels)
            loss.backward(retain_graph=True) #
            self.optimizer.step()
            
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss, total_examples

    @torch.no_grad()
    def test_on_month_data(self, year, month):
        monthly_data = self.dataset.load_monthly_data(year, month)

        new_data = monthly_data['data']
        pos_edges, pos_edge_weights, neg_edges = \
            monthly_data['accidents'], monthly_data['accident_counts'], monthly_data['neg_edges']
        
        if pos_edges is None or pos_edges.size(0) < 10:
            return {}, 0

        print(f"Eval on {year}-{month} data")
        print(f"Number of positive edges: {pos_edges.size(0)} | Number of negative edges: {neg_edges.size(0)}")
        
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
        rocauc = self.evaluator.eval(pos_preds, neg_preds)
        results.update(rocauc)

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
            mode = 'min' if (key == 'Loss' or key == "MAE" or key == "MSE") else 'max'
            train, valid, test = self.loggers[key].print_statistics(run=0, mode=mode)
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