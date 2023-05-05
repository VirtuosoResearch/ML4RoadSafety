import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric


def k_hop_neighbors(edge_index, num_nodes, num_hops):
    '''
    Find the k-hop neighbors of node indexed from 1 to num_nodes
        Assumes that flow == 'target_to_source'
    '''
    row, col = edge_index
    k_hop_nbrs = []
    for idx in range(num_nodes):
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        
        node_idx = torch.tensor([idx], device=row.device).flatten()
        subsets = [node_idx]

        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])

        subset = torch.cat(subsets).unique()
        k_hop_nbrs.append(subset)
    return k_hop_nbrs

development_seed = 1684992425

def set_train_val_test_split(seed, data, num_development = 1500, num_per_class = 20):
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data


def accuracy(output, labels):
    if len(labels.shape) != 1:
        labels = labels.squeeze()
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

@torch.no_grad()
def test_longtail_performance(model, data, test_idx, thres=8):
    # Compute test node degrees
    model.eval()
    if data.edge_index is not None:
        degrees = torch_geometric.utils.degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)
        output = model(data.x, data.edge_index)
    else:
        degrees = data.adj_t.sum(0)
        output = model(data.x, data.adj_t)
    test_degrees = degrees[test_idx]

    tmp_test_idx = test_idx[test_degrees<=thres]
    if len(tmp_test_idx) == 0:
        acc_test = -1
    else:
        acc_test = accuracy(output[tmp_test_idx], data.y[tmp_test_idx]).cpu().item()
    return acc_test

@torch.no_grad()
def test_performance_in_degrees(model, data, test_idx, degree_power=7):
    # Compute test node degrees
    model.eval()
    if data.edge_index is not None:
        degrees = torch_geometric.utils.degree(data.edge_index[1], data.x.size(0), dtype=data.x.dtype)
        output = model(data.x, data.edge_index)
    else:
        degrees = data.adj_t.sum(0)
        output = model(data.x, data.adj_t)
    test_degrees = degrees[test_idx]

    accs = []
    degree_range = np.arange(degree_power)
    for degree in np.exp2(degree_range):
        tmp_test_idx = test_idx[torch.logical_and(int(degree/2)<test_degrees, test_degrees<=degree)]
        if len(tmp_test_idx) == 0:
            accs.append(-1)
        else:
            acc_test = accuracy(output[tmp_test_idx], data.y[tmp_test_idx])
            accs.append(acc_test.cpu().item())
    return np.array(accs)
    # test_idxes_w_degrees = list(zip(test_idx, test_degrees))
    # test_idxes_w_degrees.sort(key=lambda x: x[-1])
    # sorted_test_idxes = [pair[0] for pair in test_idxes_w_degrees]
    # sorted_test_idxes = torch.LongTensor(sorted_test_idxes)

    # test_len = test_idxes.size()[0]
    # print("Long tail performance:")
    # for percent in np.arange(0.05, 1.01, 0.05):
    #     test_idx = test_idxes[:int(percent*test_len)]
    #     output = model(data.x, data.edge_index)
    #     loss_test = F.nll_loss(output[test_idx], data.y[test_idx])
    #     acc_test = accuracy(output[test_idx], data.y[test_idx])
    #     print("Test set results {:.2f}%".format(percent),
    #         "Degree= {:3.0f}".format(degrees[test_idx[-1]]),
    #         "loss= {:.4f}".format(loss_test.cpu().item()),
    #         "accuracy= {:.4f}".format(acc_test.cpu().item()))
    #     accs.append(acc_test.cpu().item())
    # return np.array(accs)