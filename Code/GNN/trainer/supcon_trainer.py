from re import S
from trainer.base_trainer import Trainer
from utils.supcon_loss import SupConLoss
import torch.nn.functional as F

class SupConTrainer(Trainer):

    def __init__(self, model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor="accuracy",
        supcon_lam = 0.9, supcon_tmp = 0.3):
        super().__init__(model, optimizer, data, split_idx, evaluator, device, 
        epochs, log_steps, checkpoint_dir, degree_thres, monitor)

        self.supcon_loss = SupConLoss(temperature=supcon_tmp)
        self.supcon_tmp = supcon_tmp
        self.supcon_lam = supcon_lam

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        if hasattr(self.data, "adj_t"):
            features = self.model(self.data.x, self.data.adj_t, return_softmax=False)[self.train_idx]
        else:
            features = self.model(self.data.x, self.data.edge_index, return_softmax=False)[self.train_idx]
        normalized_features = F.normalize(features, dim=1)
        outputs = F.log_softmax(features, dim=1)
        labels = self.data.y.squeeze(1)[self.train_idx]

        supcon_loss = self.supcon_loss(normalized_features.unsqueeze(1), labels = labels)
        ce_loss = F.nll_loss(outputs, labels)
        loss = self.supcon_lam*supcon_loss + (1-self.supcon_lam)*ce_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()