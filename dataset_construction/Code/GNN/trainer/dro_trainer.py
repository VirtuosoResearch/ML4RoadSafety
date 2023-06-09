import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np

from trainer.base_trainer import Trainer
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import to_scipy_sparse_matrix
from utils.pagerank import pagerank_scipy

class DROTrainer(Trainer):

    def __init__(self, model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor,
        group_num, weight_lr = 1, group_adj = 0, group_by="degree"):
        super().__init__(model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor)

        self.group_num = group_num # int
        self.weight_lr = weight_lr # float
        self.group_adj = group_adj # float
        self.group_labels  = self.split_groups(group_by=group_by) # size: [node_num, 1]
        self.group_weights = torch.ones((group_num, ), device=device)/group_num # size: [group_num, ]

        # record moving-average group loss
        self.gamma = 0.1
        self.exp_avg_loss = torch.zeros_like(self.group_weights, device=self.device)
        self.exp_avg_initialized = torch.zeros_like(self.group_weights, device=self.device).byte()

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss.detach().clone()*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)
    
    def split_groups(self, group_by="degree"):
        '''
        Assign group labels to nodes by:
            degrees
            pagerank values
            dispersion terms
        '''
        group_labels = torch.zeros_like(self.data.y)

        if group_by == "degree":
            if hasattr(self.data, "adj_t"):
                degrees = self.data.adj_t.sum(0)
            else:
                degrees = torch_geometric.utils.degree(self.data.edge_index[1], self.data.x.size(0), dtype=self.data.x.dtype)

            metrics = degrees
        elif group_by == "pagerank":
            ''' Compute pagerank '''
            G = to_scipy_sparse_matrix(self.data.edge_index, num_nodes=self.data.num_nodes)
            pageranks = pagerank_scipy(G, alpha=0.15)
            pageranks = torch.Tensor(pageranks).to(self.device)
            metrics = pageranks

        def split_by_median(idxes):
            tmp_median = torch.median(metrics[idxes])
            group_1 = idxes[metrics[idxes]<=tmp_median]
            group_2 = idxes[metrics[idxes]>tmp_median]
            return group_1, group_2

        group_idxes = [torch.arange(metrics.size(0))]
        for _ in range(self.group_num-1):
            tmp_idxes = group_idxes[0]
            group_1, group_2 = split_by_median(tmp_idxes)
            group_idxes.pop(0)
            group_idxes.append(group_1); group_idxes.append(group_2)
        for i, idxes in enumerate(group_idxes):
            group_labels[idxes] = i
        return group_labels.squeeze()

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        group_weights = self.group_weights

        if hasattr(self.data, "adj_t"):
            outputs = self.model(self.data.x, self.data.adj_t)[self.train_idx]
        else:
            outputs = self.model(self.data.x, self.data.edge_index)[self.train_idx]

        labels = self.data.y.squeeze(1)[self.train_idx]
        group_labels = self.group_labels[self.train_idx]

        # compute sample-wise losses and assign it to groups
        losses = F.nll_loss(outputs, labels, reduction="none")
        group_losses  = scatter_mean(losses, index=group_labels, dim_size=self.group_num, dim=0)
        group_samples = scatter_add(torch.ones_like(losses), index=group_labels, dim_size=self.group_num, dim=0)
        
        # adjust group loss with group counts
        group_losses += self.group_adj/torch.sqrt(group_samples+1)
        
        # update group weights
        group_weights = group_weights*torch.exp(self.weight_lr*group_losses.detach())
        group_weights = group_weights/torch.sum(group_weights)
        self.group_weights = group_weights
        print(f'Epoch: {epoch:02d}, ',
            f"Group weights: {group_weights}")

        # compute the total loss
        loss = torch.sum(group_losses*group_weights)
        loss.backward()
        self.optimizer.step()

        return loss.item()