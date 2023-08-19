from trainers.trainer import Trainer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils.bypass_bn import enable_running_stats, disable_running_stats

class SAMTrainer(Trainer):

    def __init__(self, model, predictor, dataset, optimizer, evaluator, 
                 train_years, valid_years, test_years, epochs, batch_size, 
                 eval_steps, device, log_metrics, use_time_series=False, input_time_steps=12):
        super().__init__(model, predictor, dataset, optimizer, evaluator, 
                         train_years, valid_years, test_years, epochs, batch_size, 
                         eval_steps, device, log_metrics, use_time_series, input_time_steps)
        
    def train_on_month_data(self, year, month): 
        monthly_data = self.dataset.load_monthly_data(year, month)

        # load previous months 
        if self.use_time_series:
            list_x = [monthly_data['data'].x]; cur_year = year; cur_month = month
            feature_dim = monthly_data['data'].x.size(1)
            for i in range(self.input_time_steps-1):
                cur_month -= 1
                if cur_month == 0:
                    cur_year -= 1
                    cur_month = 12
                prev_monthly_data = self.dataset.load_monthly_data(year, month)
                if prev_monthly_data['data'].x.shape[1] != feature_dim:
                    continue
                list_x.append(prev_monthly_data['data'].x)
            inputs = torch.stack(list_x, dim=0).unsqueeze(0)
        else:
            inputs = monthly_data['data'].x

        new_data = monthly_data['data']
        pos_edges, pos_edge_weights, neg_edges = \
            monthly_data['accidents'], monthly_data['accident_counts'], monthly_data['neg_edges']
        
        if pos_edges is None or pos_edges.size(0) < 10:
            return 0, 0
        
        self.model.train()
        self.predictor.train()

        '''
        First forward-backward step
        '''
        enable_running_stats(self.model)

        # encoding
        new_data = new_data.to(self.device); inputs = inputs.to(self.device)
        edge_attr = new_data.edge_attr
        h = self.model(inputs, new_data.edge_index, edge_attr)
        if len(h.size()) == 4:
            h = h.squeeze(0)[-1, :, :]
        if len(h.size()) == 3:
            h = h[-1, :, :]

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
            # self.optimizer.step()
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        self.optimizer.first_step(zero_grad=True)

        '''
        Second forward-backward step
        '''
        disable_running_stats(self.model)        
        # encoding
        new_data = new_data.to(self.device); inputs = inputs.to(self.device)
        edge_attr = new_data.edge_attr
        h = self.model(inputs, new_data.edge_index, edge_attr)
        if len(h.size()) == 4:
            h = h.squeeze(0)[-1, :, :]
        if len(h.size()) == 3:
            h = h[-1, :, :]

        # predicting
        pos_train_edge = pos_edges.to(self.device)
        pos_edge_weights = pos_edge_weights.to(self.device)
        neg_edges = neg_edges.to(self.device)
        # self.batch_size > pos_train_edge.size(0): only backprop once since it does not retain cache.
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
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
            self.evaluator.criterion(torch.cat([pos_out, neg_out]), labels).backward()
        self.optimizer.second_step(zero_grad=True)
        
        return total_loss, total_examples