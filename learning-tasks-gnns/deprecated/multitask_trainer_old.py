from trainers.trainer import Trainer
from data_loaders import load_monthly_data, load_yearly_data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from evaluators import eval_mae, eval_hits, eval_rocauc
from logger import Logger
from torch_geometric.loader import NeighborLoader

class MultitaskTrainer(Trainer):

    def __init__(self, model, predictor, data, optimizer, data_dir, state_name, 
                 train_years, valid_years, test_years, epochs, batch_size, eval_steps, device, 
                 log_metrics=[], 
                 use_dynamic_node_features=False, 
                 use_dynamic_edge_features=False, 
                 num_negative_edges=10000, 
                 node_feature_mean=None, node_feature_std=None, edge_feature_mean=None, edge_feature_std=None, 
                 if_sample_node=False, sample_batch_size=4098,
                 tasks={}, task_to_datas={}, task_to_predictors={}, task_to_metrics=[]):
        super().__init__(model, predictor, data, optimizer, data_dir, state_name, 
                train_years, valid_years, test_years, epochs, batch_size, eval_steps, device, 
                log_metrics, use_dynamic_node_features, use_dynamic_edge_features, num_negative_edges, 
                node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std, 
                if_sample_node, sample_batch_size)
        
        self.tasks = tasks
        self.task_to_datas = task_to_datas
        self.task_to_predictors = task_to_predictors
        self.task_to_metrics = task_to_metrics

        self.loggers = {}
        for task_name in self.tasks:
            self.loggers.update({
                key: Logger(runs=1) for key in self.task_to_metrics[task_name]
            })

        self.task_to_train_years = {}
        self.task_to_valid_years = {}
        self.task_to_test_years = {}

    def train_on_month_data(self, year, month, task_name, task_type): 
        data = self.task_to_datas[task_name]
        state_name = task_name.split("_")[0]
        predictor = self.task_to_predictors[task_name]

        pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(data, data_dir=self.data_dir, state_name=state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)

        if pos_edges is None or pos_edges.size(0) == 0:
            return 0, 0

        # normalize node and edge features
        if self.node_feature_mean is not None:
            node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        if self.edge_feature_mean is not None:
            edge_features = (edge_features - self.edge_feature_mean) / self.edge_feature_std

        new_data = data.clone()
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
        predictor.train()

        # encoding
        new_data = new_data.to(self.device)
        h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr

        # predicting
        pos_train_edge = pos_edges.to(self.device)
        total_loss = total_examples = 0
        # self.batch_size > pos_train_edge.size(0): only backprop once since it does not retain cache.
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
            self.optimizer.zero_grad()
            # positive edges
            edge = pos_train_edge[perm].t()
            pos_out = predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                predictor(h[edge[0]], h[edge[1]], edge_attr[perm])

            # sampling from negative edges
            neg_masks = np.random.choice(neg_edges.size(0), min(edge.size(1), neg_edges.size(0)), replace=False)
            edge = neg_edges[neg_masks].t() # torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=device)
            neg_out = predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            
            if task_type == "regression":
                labels = torch.cat([pos_edge_weights[perm].view(-1, 1), torch.zeros_like(neg_out)]).view(-1, 1).to(self.device)
                loss = F.l1_loss(torch.cat([pos_out, neg_out]), labels)
            else:
                labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(self.device)
                loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
            loss.backward(retain_graph=True) # 
            self.optimizer.step()
            
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss, total_examples
    
    def train_on_year_data(self, year, task_name): 
        data = self.task_to_datas[task_name]
        state_name = task_name.split("_")[0]
        predictor = self.task_to_predictors[task_name]

        pos_edges, pos_edge_weights, node_features = load_yearly_data(data_dir=self.data_dir, state_name=state_name, year=year)
        
        # normalize node and edge features
        if self.node_feature_mean is not None:
            node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        
        if pos_edges.size(0) == 0:
            return 0, 0

        new_data = data.clone()
        if self.use_dynamic_node_features:
            if new_data.x is None:
                new_data.x = node_features
            else:
                new_data.x = torch.cat([new_data.x, node_features], dim=1)
        
        self.model.train()
        predictor.train()

        # encoding
        new_data = new_data.to(self.device)
        h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr

        # predicting
        pos_edge_weights = pos_edge_weights.to(self.device)
        pos_train_edge = pos_edges.to(self.device)
        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
            self.optimizer.zero_grad()
            # positive edges
            edge = pos_train_edge[perm].t()
            pos_out = predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            
            labels = pos_edge_weights.view(-1, 1).to(self.device)
            loss = F.l1_loss(pos_out, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss, total_examples
        
    def train_epoch(self, task_name = "MA_accident_classification"):
        total_loss = total_examples = 0

        state_name, data_type, task_type = task_name.split("_")
        train_years = self.task_to_train_years[task_name]
        for year in train_years:
            if data_type == "accident":
                for month in range(1, 13):
                    loss, samples = self.train_on_month_data(year, month, task_name = task_name, task_type=task_type)
                    total_loss += loss
                    total_examples += samples
            elif data_type == "volume":
                loss, samples = self.train_on_year_data(year, task_name = task_name)
                total_loss += loss
                total_examples += samples
        return total_loss/total_examples

    def train(self):
        train_log = {}
        for epoch in range(1, 1 + self.epochs):
            task_list = self.tasks[:]
            np.random.shuffle(task_list)

            losses = {}
            for task_name in task_list:
                loss = self.train_epoch(task_name=task_name)
                losses[task_name] = loss

            if epoch % self.eval_steps == 0:
                for task_name in self.tasks:
                    results = self.test(task_name)
                    results = {f"{task_name}_{metric_name}": result for metric_name, result in results.items()}

                    for key, result in results.items():
                        self.loggers[key].add_result(run=0, result=result)
                
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Epoch: {epoch:02d}, '
                            f'Loss: {losses[task_name]:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in self.loggers.keys():
            print(key)
            mode = 'min' if ('Loss' in key or "MAE" in key or "MSE" in key) else 'max'
            train, valid, test = self.loggers[key].print_statistics(run=0, mode=mode)
            train_log[f"Train_{key}"] = train
            train_log[f"Valid_{key}"] = valid
            train_log[f"Test_{key}"] = test
        return train_log


    @torch.no_grad()
    def test_on_month_data(self, year, month, task_name, task_type):
        data = self.task_to_datas[task_name]
        state_name = task_name.split("_")[0]
        predictor = self.task_to_predictors[task_name]

        pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(data, data_dir=self.data_dir, state_name=state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)

        if pos_edges is None or pos_edges.size(0) == 0:
            return {}, 0

        print(f"Eval on {year}-{month} data")
        print(f"Number of positive edges: {pos_edges.size(0)} | Number of negative edges: {neg_edges.size(0)}")

        # normalize node and edge features
        if self.node_feature_mean is not None:
            node_features = (node_features - self.node_feature_mean) / self.node_feature_std
        if self.edge_feature_mean is not None:
            edge_features = (edge_features - self.edge_feature_mean) / self.edge_feature_std

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
        predictor.eval()

        # encoding
        if not self.if_sample_node:
            new_data = new_data.to(self.device)
            h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
            edge_attr = new_data.edge_attr
        else:
            train_loader = NeighborLoader(new_data, 
                                          num_neighbors=[-1]*self.model.num_layer, 
                                          batch_size=self.sample_batch_size)
            h = []
            for batch in train_loader:
                batch = batch.to(self.device)
                batch_h = self.model(batch.x, batch.edge_index, batch.edge_attr)
                h.append(batch_h[:batch.batch_size])
            h = torch.cat(h, dim=0)
            edge_attr = new_data.edge_attr

        # predicting
        pos_edge = pos_edges.to(self.device)
        neg_edge = neg_edges.to(self.device)
        pos_preds = []
        for perm in DataLoader(range(pos_edge.size(0)), self.batch_size):
            edge = pos_edge[perm].t()
            preds = predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            pos_preds += [preds.squeeze().cpu()] 
        pos_preds = torch.cat(pos_preds, dim=0)

        neg_preds = []
        for perm in DataLoader(range(neg_edge.size(0)), self.batch_size):
            edge = neg_edge[perm].t()
            preds = predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            neg_preds += [preds.squeeze().cpu()]
        neg_preds = torch.cat(neg_preds, dim=0)

        results = {}

        if task_type == "regression":
            pos_edge_weights = pos_edge_weights.cpu()
            results = eval_mae(torch.cat([pos_preds, neg_preds], dim=0), torch.cat([pos_edge_weights, torch.zeros_like(neg_preds)], dim=0))
        else:
            # Eval ROC-AUC
            rocauc = eval_rocauc(pos_preds, neg_preds)
            results.update(rocauc)
            # Eval Hits@K
            hits = eval_hits(pos_preds, neg_preds, K=100)
            results.update(hits)

        return results, pos_edges.size(0)


    @torch.no_grad()
    def test_on_year_data(self, year, task_name):
        data = self.task_to_datas[task_name]
        state_name = task_name.split("_")[0]
        predictor = self.task_to_predictors[task_name]

        pos_edges, pos_edge_weights, node_features = load_yearly_data(data_dir=self.data_dir, state_name=state_name, year=year)

        print(f"Eval on {year} data")
        print(f"Number of edges with valid traffic volume: {pos_edges.size(0)}")

        # normalize node and edge features
        if self.node_feature_mean is not None:
            node_features = (node_features - self.node_feature_mean) / self.node_feature_std
       
        if pos_edges.size(0) == 0:
            return {}, 0

        new_data = data.clone()
        if self.use_dynamic_node_features:
            if new_data.x is None:
                new_data.x = node_features
            else:
                new_data.x = torch.cat([new_data.x, node_features], dim=1)

        self.model.eval()
        predictor.eval()

        # encoding
        new_data = new_data.to(self.device)
        h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr

        # predicting
        pos_edge = pos_edges.to(self.device)
        pos_preds = []
        for perm in DataLoader(range(pos_edge.size(0)), self.batch_size):
            edge = pos_edge[perm].t()
            preds = predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            pos_preds += [preds.squeeze().cpu()] 
        pos_preds = torch.cat(pos_preds, dim=0)

        # Eval ROC-AUC
        results = eval_mae(pos_preds, pos_edge_weights)

        return results, pos_edges.size(0)


    def test(self, task_name):
        state_name, data_type, task_type = task_name.split("_")
        train_years = self.task_to_train_years[task_name]
        valid_years = self.task_to_valid_years[task_name]
        test_years = self.task_to_test_years[task_name]

        if data_type == "accident":
            train_results = {}; train_size = 0
            for year in train_years:
                for month in range(1, 13):
                    month_results, month_sample_size = self.test_on_month_data(year, month, task_name, task_type)
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
                    month_results, month_sample_size = self.test_on_month_data(year, month, task_name, task_type)
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
                    month_results, month_sample_size = self.test_on_month_data(year, month, task_name, task_type)
                    for key, value in month_results.items():
                        if key not in test_results:
                            test_results[key] = 0
                        test_results[key] += value * month_sample_size
                    test_size += month_sample_size
                
            for key, value in test_results.items():
                test_results[key] = value / test_size
        else:
            train_results = {}; train_size = 0
            for year in train_years:
                month_results, month_sample_size = self.test_on_year_data(year, task_name)
                for key, value in month_results.items():
                    if key not in train_results:
                        train_results[key] = 0
                    train_results[key] += value * month_sample_size
                train_size += month_sample_size

            for key, value in train_results.items():
                train_results[key] = value / train_size

            val_results = {}; val_size = 0
            for year in valid_years:
                month_results, month_sample_size = self.test_on_year_data(year, task_name)
                for key, value in month_results.items():
                    if key not in val_results:
                        val_results[key] = 0
                    val_results[key] += value * month_sample_size
                val_size += month_sample_size
            
            for key, value in val_results.items():
                val_results[key] = value / val_size

            test_results = {}; test_size = 0
            for year in test_years:
                month_results, month_sample_size = self.test_on_year_data(year, task_name)
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