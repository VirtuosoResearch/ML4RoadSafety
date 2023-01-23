import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.base_trainer import Trainer
from utils.constraint import FrobeniusConstraint, LInfLipschitzConstraint

class ConstraintTrainer(Trainer):

    def __init__(self, model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor="accuracy"):
        super().__init__(model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor)

        self.constraints = []
    
    def add_constraint(self, lambda_extractor, norm='frob', state_dict = None):
        '''
        Add hard constraint for model weights
            for feature_extractor, it will contraint the weight to pretrain weight
            for pred_head, it will contraint the weight to zero
        '''
        
        # is not use_ratio, then both the lambda_extractor & lambda_pred_head is absolute value; 
        # here we could use layer-wise distance
        if norm == "inf-op":
            self.constraints.append(
                LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                state_dict = state_dict, excluding_key = "bn")
            )
        elif norm == "frob":
            self.constraints.append(
                FrobeniusConstraint(type(self.model), lambda_extractor, 
                state_dict = state_dict, excluding_key = "bn")
            )

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        if hasattr(self.data, "adj_t"):
            outputs = self.model(self.data.x, self.data.adj_t)[self.train_idx]
        else:
            outputs = self.model(self.data.x, self.data.edge_index)[self.train_idx]

        labels = self.data.y.squeeze(1)[self.train_idx]

        loss = F.nll_loss(outputs, labels)
        loss.backward()
        self.optimizer.step()

        """Apply Constraints"""
        for constraint in self.constraints:
            self.model.apply(constraint)
        """Apply Constraints"""

        return loss.item()
