import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bypass_bn import disable_running_stats, enable_running_stats
from trainer.base_trainer import Trainer

class SAMTrainer(Trainer):

    def __init__(self, model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor="accuracy"):
        super().__init__(model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        # first forward-backward step
        enable_running_stats(self.model)
        if hasattr(self.data, "adj_t"):
            outputs = self.model(self.data.x, self.data.adj_t)[self.train_idx]
        else:
            outputs = self.model(self.data.x, self.data.edge_index)[self.train_idx]

        labels = self.data.y.squeeze(1)[self.train_idx]

        loss = F.nll_loss(outputs, labels)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(self.model)
        if hasattr(self.data, "adj_t"):
            F.nll_loss(
                self.model(self.data.x, self.data.adj_t)[self.train_idx], labels
            ).backward()
        else:
            F.nll_loss(
                self.model(self.data.x, self.data.edge_index)[self.train_idx], labels
            ).backward()
        self.optimizer.second_step(zero_grad=True)
        return loss.item()