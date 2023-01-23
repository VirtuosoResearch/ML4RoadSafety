import torch
import model
from torch.nn import BatchNorm2d

def _frob_norm(w):
    return torch.sum(torch.pow(w, 2.0))

def _linf_norm(w):
    if len(w.shape) != 2 and len(w.shape) != 4:
        assert len(w.shape) == 1
        return torch.max(torch.abs(w))
    
    axes=1

    if len(w.shape) == 4:
        axes=[1, 2, 3]
    
    norm = torch.max(torch.sum(torch.abs(w), dim=axes))
    return norm

class LInfLipschitzConstraint(object): 

    def __init__(self, model_type, max_k, state_dict = None, 
                 excluding_key = None, including_key = None) -> None:
        self.model_type = model_type
        self.max_k = max_k
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            for param in module.named_parameters():
                name = param[0]
                if "bias" in name:
                    continue
                
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                w = param[1].data 
                if len(w.shape) != 2 and len(w.shape)!= 4:
                    continue

                if self.state_dict is not None:
                    t = w - self.state_dict[param[0]]
                else:
                    t = w

                axes=1
                
                if len(w.shape) == 4:
                    axes=[1, 2, 3]

                norms = torch.sum(torch.abs(t), dim=axes, keepdims=True)
                v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=norms.device), norms / self.max_k))
                
                if self.state_dict is not None:
                    param[1].data = self.state_dict[param[0]] + v
                else:
                    param[1].data = v
            for name, sub_module in module.named_modules():
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                if type(sub_module) == BatchNorm2d:
                    weight = sub_module.weight

                    w = sub_module.weight.data
                    variance = sub_module.running_var.data
                    if self.state_dict is not None:
                        zero_weight = self.state_dict[name+".weight"]
                        zero_variance = self.state_dict[name+".running_var"]

                    diag = w / torch.sqrt(variance + 1e-6)

                    if self.state_dict is not None:
                        zero_diag = (zero_weight / torch.sqrt(zero_variance  + 1e-6))
                        t = diag - zero_diag
                    else:
                        t = diag

                    v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=weight.device), torch.abs(t) / self.max_k))
                    
                    if self.state_dict is not None:
                        weight.data =  (v + zero_diag) * torch.sqrt(variance + 1e-6)
                    else:
                        weight.data =  v * torch.sqrt(variance + 1e-6)


class FrobeniusConstraint(object): 

    def __init__(self, model_type, max_k, state_dict = None,
                 excluding_key = None, including_key = None) -> None:
        self.model_type = model_type
        self.max_k = max_k
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            for param in module.named_parameters():
                name = param[0]
                if "bias" in name:
                    continue
            
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                w = param[1].data 
                if len(w.shape) != 2 and len(w.shape)!= 4:
                    continue

                if self.state_dict is not None:
                    t = w - self.state_dict[param[0]]
                else:
                    t = w

                norm = torch.sqrt(_frob_norm(t))
                v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=norm.device), norm / self.max_k))
                
                if self.state_dict is not None:
                    param[1].data = self.state_dict[param[0]] + v
                else:
                    param[1].data = v
            for name, sub_module in module.named_modules():
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                if type(sub_module) == BatchNorm2d:
                    weight = sub_module.weight

                    w = sub_module.weight.data
                    variance = sub_module.running_var.data
                    if self.state_dict is not None:
                        zero_weight = self.state_dict[name+".weight"]
                        zero_variance = self.state_dict[name+".running_var"]

                    diag = w / torch.sqrt(variance + 1e-6)

                    if self.state_dict is not None:
                        zero_diag = (zero_weight / torch.sqrt(zero_variance  + 1e-6))
                        t = diag - zero_diag
                    else:
                        t = diag

                    v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=weight.device), torch.abs(t) / self.max_k))
                    
                    if self.state_dict is not None:
                        weight.data =  (v + zero_diag) * torch.sqrt(variance + 1e-6)
                    else:
                        weight.data =  v * torch.sqrt(variance + 1e-6)

def _create_penalty(weight, name, _lambda, dist_func, state_dict=None):
    if state_dict is not None:
        return _lambda * dist_func(weight - state_dict[name])
    else:
        return _lambda * dist_func(weight)

def add_penalty(model, norm, _lambda, excluding_key = None, including_key = None, state_dict=None):
    penalty = 0
    for param in model.named_parameters():
        name = param[0]
        if "bias" in name:
            continue
        if (excluding_key is not None) and (excluding_key in name):
            continue
        if (including_key is not None) and (including_key not in name):
            continue
        w = param[1]

        if norm == "inf-op":
            penalty += _create_penalty(w, name, _lambda, _linf_norm, state_dict)
        elif norm == "frob":
            penalty += _create_penalty(w, name, _lambda, _frob_norm, state_dict)
    return penalty

class LInfLipschitzConstraintRatio(object): 

    def __init__(self, model_type, max_ratio, state_dict = None, 
                 excluding_key = None, including_key = None) -> None:
        self.model_type = model_type
        self.max_ratio = max_ratio
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            for param in module.named_parameters():
                name = param[0]
                if "bias" in name:
                    continue
                
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                w = param[1].data 
                if len(w.shape) != 2 and len(w.shape)!= 4:
                    continue

                if self.state_dict is not None:
                    t = w - self.state_dict[param[0]]
                else:
                    t = w

                axes=1
                
                if len(w.shape) == 4:
                    axes=[1, 2, 3]

                norms = torch.sum(torch.abs(t), dim=axes, keepdims=True)
                tmp_max_k = torch.sum(torch.abs(self.state_dict[param[0]]), dim=axes, keepdims=True) * self.max_ratio
                v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=norms.device), norms / tmp_max_k))
                
                if self.state_dict is not None:
                    param[1].data = self.state_dict[param[0]] + v
                else:
                    param[1].data = v
            for name, sub_module in module.named_modules():
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                if type(sub_module) == BatchNorm2d:
                    weight = sub_module.weight

                    w = sub_module.weight.data
                    variance = sub_module.running_var.data
                    if self.state_dict is not None:
                        zero_weight = self.state_dict[name+".weight"]
                        zero_variance = self.state_dict[name+".running_var"]

                    diag = w / torch.sqrt(variance + 1e-6)

                    if self.state_dict is not None:
                        zero_diag = (zero_weight / torch.sqrt(zero_variance  + 1e-6))
                        t = diag - zero_diag
                    else:
                        t = diag

                    tmp_max_k = torch.abs(zero_diag) * self.max_ratio
                    v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=weight.device), torch.abs(t) / tmp_max_k))
                    
                    if self.state_dict is not None:
                        weight.data =  (v + zero_diag) * torch.sqrt(variance + 1e-6)
                    else:
                        weight.data =  v * torch.sqrt(variance + 1e-6)

class FrobeniusConstraintRatio(object): 

    def __init__(self, model_type, max_ratio, state_dict = None,
                 excluding_key = None, including_key = None) -> None:
        self.model_type = model_type
        self.max_ratio = max_ratio
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            for param in module.named_parameters():
                name = param[0]
                if "bias" in name:
                    continue
            
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                w = param[1].data 
                if len(w.shape) != 2 and len(w.shape)!= 4:
                    continue

                if self.state_dict is not None:
                    t = w - self.state_dict[name]
                else:
                    t = w

                norm = torch.sqrt(_frob_norm(t))
                tmp_max_k = torch.sqrt(_frob_norm(self.state_dict[name])) * self.max_ratio
                v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=norm.device), norm / tmp_max_k))
                
                if self.state_dict is not None:
                    param[1].data = self.state_dict[name] + v
                else:
                    param[1].data = v
            for name, sub_module in module.named_modules():
                if (self.excluding_key is not None) and (self.excluding_key in name):
                    continue
                if (self.including_key is not None) and (self.including_key not in name):
                    continue
                if type(sub_module) == BatchNorm2d:
                    weight = sub_module.weight

                    w = sub_module.weight.data
                    variance = sub_module.running_var.data
                    if self.state_dict is not None:
                        zero_weight = self.state_dict[name+".weight"]
                        zero_variance = self.state_dict[name+".running_var"]

                    diag = w / torch.sqrt(variance + 1e-6)

                    if self.state_dict is not None:
                        zero_diag = (zero_weight / torch.sqrt(zero_variance  + 1e-6))
                        t = diag - zero_diag
                    else:
                        t = diag

                    tmp_max_k = torch.abs(zero_diag) * self.max_ratio
                    v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=weight.device), torch.abs(t) / tmp_max_k))
                    
                    if self.state_dict is not None:
                        weight.data =  (v + zero_diag) * torch.sqrt(variance + 1e-6)
                    else:
                        weight.data =  v * torch.sqrt(variance + 1e-6)

class FrobeniusConstraintRatioList(object): 

    def __init__(self, model_type, ratios, names, state_dict = None) -> None:
        self.model_type = model_type
        assert len(ratios) == len(names)
        self.ratios = {}
        self.names = names
        for i, name in enumerate(names):
            self.ratios[name] = ratios[i]

        self.state_dict = state_dict

    def __call__(self, module):
        if type(module) == self.model_type:
            for param in module.named_parameters():
                name = param[0]
                if "bias" in name:
                    continue
                if name not in self.names:
                    continue
                
                w = param[1].data 
                if len(w.shape) != 2 and len(w.shape)!= 4:
                    continue

                if self.state_dict is not None:
                    t = w - self.state_dict[name]
                else:
                    t = w

                norm = torch.sqrt(_frob_norm(t))
                tmp_max_k = torch.sqrt(_frob_norm(self.state_dict[name])) * self.ratios[name]
                v = t * (1.0 / torch.maximum(torch.tensor(1.0, device=norm.device), norm / tmp_max_k))
                
                if self.state_dict is not None:
                    param[1].data = self.state_dict[name] + v
                else:
                    param[1].data = v