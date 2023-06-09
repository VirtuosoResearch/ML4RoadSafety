from trainers.trainer import Trainer
from data_loaders import load_monthly_data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from evaluators import eval_mae

class AccidentRegressionTrainer(Trainer):

    def __init__(self, model, predictor, data, optimizer, 
                 data_dir, state_name, train_years, valid_years, test_years, 
                 epochs, batch_size, eval_steps, device, 
                 log_metrics=[], 
                 use_dynamic_node_features=False, 
                 use_dynamic_edge_features=False, 
                 num_negative_edges=10000, 
                 node_feature_mean=None, 
                 node_feature_std=None, 
                 edge_feature_mean=None, 
                 edge_feature_std=None):
        super().__init__(model, predictor, data, optimizer, 
                         data_dir, state_name, train_years, valid_years, test_years, 
                         epochs, batch_size, eval_steps, device, 
                         log_metrics, use_dynamic_node_features, use_dynamic_edge_features, num_negative_edges, 
                         node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std)
    
    def train_on_month_data(self, year, month): 
        pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(self.data, data_dir=self.data_dir, state_name=self.state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)

        if pos_edges is None or pos_edges.size(0) < 10:
            return 0, 0

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
        
        self.model.train()
        self.predictor.train()

        # encoding
        new_data = new_data.to(self.device)
        h = self.model(new_data.x, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr

        # predicting
        pos_train_edge = pos_edges.to(self.device)
        pos_edge_weights = pos_edge_weights.to(self.device)
        total_loss = total_examples = 0
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
            labels = torch.cat([pos_edge_weights[perm].view(-1, 1), torch.zeros_like(neg_out)]).view(-1, 1).to(self.device)

            loss = F.l1_loss(torch.cat([pos_out, neg_out]), labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            num_examples = pos_out.size(0) + neg_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss, total_examples

    @torch.no_grad()
    def test_on_month_data(self, year, month):
        pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = \
            load_monthly_data(self.data, data_dir=self.data_dir, state_name=self.state_name, year=year, month = month, num_negative_edges=self.num_negative_edges)

        if pos_edges is None or pos_edges.size(0) < 10:
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

        pos_edge_weights = pos_edge_weights.cpu()
        results = eval_mae(torch.cat([pos_preds, neg_preds], dim=0), torch.cat([pos_edge_weights, torch.zeros_like(neg_preds)], dim=0))

        return results, pos_edges.size(0)+neg_edges.size(0)