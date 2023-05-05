from torch_geometric.utils import dropout_adj


def uniform_sampling_edges(data, p = 0.5):
    data.edge_index, _ = dropout_adj(
        data.edge_index, p=p, force_undirected=True, num_nodes=data.num_nodes)
    return data
