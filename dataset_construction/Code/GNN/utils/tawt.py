import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def get_average_feature_gradients(model, loss):
    feature_gradients = grad(loss, model.parameters(), retain_graph=True, create_graph=False,
                             allow_unused=True)
    feature_gradients = torch.cat([gradient.view(-1) for gradient in feature_gradients]) # flatten gradients
    return feature_gradients

def get_task_weights_gradients_multi(model, losses, target_loss, device):
    feature_gradients = []
    for i in range(losses.size(0)):
        task_gradients = get_average_feature_gradients(model, losses[i])
        feature_gradients.append(task_gradients)
    target_gradient = get_average_feature_gradients(model, target_loss)

    num_tasks = losses.size(0)
    task_weights_gradients = torch.zeros((num_tasks, ), device=device, dtype=torch.float)
    for i, tmp_gradients in enumerate(feature_gradients):
        task_weights_gradients[i] = -F.cosine_similarity(target_gradient, tmp_gradients, dim=0)
    return task_weights_gradients