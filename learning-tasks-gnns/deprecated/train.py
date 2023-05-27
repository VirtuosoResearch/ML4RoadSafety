def train(predictor, x, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(x.device)

        loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, evaluator, batch_size):
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    # Eval ROC-AUC
    train_rocauc = eval_rocauc(pos_train_pred, neg_valid_pred)
    valid_rocauc = eval_rocauc(pos_valid_pred, neg_valid_pred)
    test_rocauc = eval_rocauc(pos_test_pred, neg_test_pred)
    results['ROC-AUC'] = (train_rocauc['ROC-AUC'], valid_rocauc['ROC-AUC'], test_rocauc['ROC-AUC'])
    results['F1'] = (train_rocauc["F1"], valid_rocauc["F1"], test_rocauc["F1"])
    results['AP'] = (train_rocauc["AP"], valid_rocauc["AP"], test_rocauc["AP"])
    
    for K in [1, 3, 5, 10]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

# dataset = PygLinkPropPredDataset(name='ogbl-collab')
# split_edge = dataset.get_edge_split()
# data = dataset[0]

# data, split_edge = load_network_with_accidents(data_dir="./data", state_name="MA", 
#                                             #    train_years=[2002], train_months=[1, 2, 3, 4],
#                                             #    valid_years=[2002], valid_months=[5, 6, 7, 8],
#                                             #    test_years=[2002], test_months=[9, 10, 11, 12],
#                                             num_negative_edges = 1000000,
#                                             feature_type="verse", feature_name = "MA_ppr_128.npy")
# print("Number of nodes: {} Number of edges: {}".format(data.num_nodes, data.num_edges))
# print("Training edges {} Validation edges {} Test edges {}".format(split_edge['train']['edge'].shape[0],
#                                                                    split_edge['valid']['edge'].shape[0],
#                                                                    split_edge['test']['edge'].shape[0]))
# x = data.x
# x = x.to(device)



def compute_feature_mean_std(data, data_dir, state_name, years):
    all_node_features = []
    all_edge_features = []
    for year in years:
        for month in range(1, 13):
            _, _, _, node_features, edge_features = load_monthly_data(data, data_dir=data_dir, state_name=state_name, year=year, month = month, num_negative_edges=10000)
            all_node_features.append(node_features)
            all_edge_features.append(edge_features)
        
    all_node_features = torch.cat(all_node_features, dim=0)
    all_edge_features = torch.cat(all_edge_features, dim=0)

    node_feature_mean, node_feature_std = all_node_features.mean(dim=0), all_node_features.std(dim=0)
    edge_feature_mean, edge_feature_std = all_edge_features.mean(dim=0), all_edge_features.std(dim=0)
    return node_feature_mean, node_feature_std, edge_feature_mean, edge_feature_std

def train_on_month_data(data, predictor, optimizer, batch_size, year, month, device, 
                        add_node_features = False, add_edge_features = False, 
                        node_feature_mean = None, node_feature_std = None, 
                        edge_feature_mean = None, edge_feature_std = None): 
    pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = load_monthly_data(data, data_dir="./data", state_name="MA", year=year, month = month, num_negative_edges=10000)

    if node_feature_mean is not None:
        node_features = (node_features - node_feature_mean) / node_feature_std
    if edge_feature_mean is not None:
        edge_features = (edge_features - edge_feature_mean) / edge_feature_std

    if pos_edges.size(0) == 0:
        return 0, 0

    new_data = data.clone()
    if add_node_features:
        if new_data.x is None:
            new_data.x = node_features
        else:
            new_data.x = torch.cat([new_data.x, node_features], dim=1)

    if add_edge_features:
        if new_data.edge_attr is None:
            new_data.edge_attr = edge_features
        else:
            new_data.edge_attr = torch.cat([new_data.edge_attr, edge_features], dim=1)
    
    predictor.train()
    x = new_data.x.to(device)
    edge_attr = new_data.edge_attr.to(device) if new_data.edge_attr is not None else None
    pos_train_edge = pos_edges.to(device)

    '''
    Use GCN Encoder 
    '''

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        edge = pos_train_edge[perm].t()
        pos_out = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])

        # Sampling from negative edges
        neg_masks = np.random.choice(neg_edges.size(0), edge.size(1), replace=False)
        edge = neg_edges[neg_masks].t() # torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=device)
        neg_out = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).view(-1, 1).to(device)

        loss = F.binary_cross_entropy(torch.cat([pos_out, neg_out]), labels)
        loss.backward()
        optimizer.step()
        
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    return total_loss, total_examples

@torch.no_grad()
def test_on_month_data(data, predictor, batch_size, year, month, device, 
                        add_node_features = False, add_edge_features = False,
                        node_feature_mean = None, node_feature_std = None,
                        edge_feature_mean = None, edge_feature_std = None):
    pos_edges, pos_edge_weights, neg_edges, node_features, edge_features = load_monthly_data(data, data_dir="./data", state_name="MA", year=year, month = month, num_negative_edges=10000)

    print(f"Eval on {year}-{month} data")
    print(f"Number of positive edges: {pos_edges.size(0)} | Number of negative edges: {neg_edges.size(0)}")

    if node_feature_mean is not None:
        node_features = (node_features - node_feature_mean) / node_feature_std
    if edge_feature_mean is not None:
        edge_features = (edge_features - edge_feature_mean) / edge_feature_std

    if pos_edges.size(0) == 0:
        return {}, 0

    new_data = data.clone()
    if add_node_features:
        if new_data.x is None:
            new_data.x = node_features
        else:
            new_data.x = torch.cat([new_data.x, node_features], dim=1)
    
    if add_edge_features:
        if new_data.edge_attr is None:
            new_data.edge_attr = edge_features
        else:
            new_data.edge_attr = torch.cat([new_data.edge_attr, edge_features], dim=1)
    
    predictor.eval()

    x = new_data.x.to(device)
    edge_attr = new_data.edge_attr.to(device) if new_data.edge_attr is not None else None
    pos_edge = pos_edges.to(device)
    neg_edge = neg_edges.to(device)

    pos_preds = []
    for perm in DataLoader(range(pos_edge.size(0)), batch_size):
        edge = pos_edge[perm].t()
        preds = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])
        pos_preds += [preds.squeeze().cpu()] 
    pos_preds = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for perm in DataLoader(range(neg_edge.size(0)), batch_size):
        edge = neg_edge[perm].t()
        preds = predictor(x[edge[0]], x[edge[1]]) if edge_attr is None else predictor(x[edge[0]], x[edge[1]], edge_attr[perm])
        neg_preds += [preds.squeeze().cpu()]
    neg_preds = torch.cat(neg_preds, dim=0)

    results = {}
    # Eval ROC-AUC
    rocauc = eval_rocauc(pos_preds, neg_preds)
    results.update(rocauc)

    hits = eval_hits(pos_preds, neg_preds, K=100)
    results.update(hits)

    return results, pos_edges.size(0)

def train(data, predictor, optimizer, batch_size, device, years, 
        add_node_features = False, add_edge_features = False,
        node_feature_mean = None, node_feature_std = None,
        edge_feature_mean = None, edge_feature_std = None):
    total_loss = total_examples = 0
    for year in years:
        for month in range(1, 13):
            loss, samples = train_on_month_data(data, predictor, optimizer, batch_size, year, month, device, 
                                                add_node_features, add_edge_features,
                                                node_feature_mean, node_feature_std,
                                                edge_feature_mean, edge_feature_std)
            total_loss += loss
            total_examples += samples
    return total_loss/total_examples

def test(data, predictor, batch_size, train_years, valid_years, test_years, device, 
         add_node_features = False, add_edge_features = False,
         node_feature_mean = None, node_feature_std = None,
         edge_feature_mean = None, edge_feature_std = None):
    train_results = {}; train_size = 0
    for year in train_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, batch_size, year, month, device, 
                                                                  add_node_features, add_edge_features,
                                                                  node_feature_mean, node_feature_std,
                                                                  edge_feature_mean, edge_feature_std)
            for key, value in month_results.items():
                if key not in train_results:
                    train_results[key] = 0
                train_results[key] += value * month_sample_size
            train_size += month_sample_size

    for key, value in train_results.items():
        train_results[key] = value / train_size

    val_results = {}; val_size = 0
    for year in valid_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, batch_size, year, month, device, 
                                                                  add_node_features, add_edge_features,
                                                                  node_feature_mean, node_feature_std,
                                                                  edge_feature_mean, edge_feature_std)
            for key, value in month_results.items():
                if key not in val_results:
                    val_results[key] = 0
                val_results[key] += value * month_sample_size
            val_size += month_sample_size
    
    for key, value in val_results.items():
        val_results[key] = value / val_size

    test_results = {}; test_size = 0
    for year in test_years:
        for month in range(1, 13):
            month_results, month_sample_size = test_on_month_data(data, predictor, batch_size, year, month, device, 
                                                                  add_node_features, add_edge_features,
                                                                  node_feature_mean, node_feature_std,
                                                                  edge_feature_mean, edge_feature_std)
            for key, value in month_results.items():
                if key not in test_results:
                    test_results[key] = 0
                test_results[key] += value * month_sample_size
            test_size += month_sample_size
        
    for key, value in test_results.items():
        test_results[key] = value / test_size

    results = {}
    for key in train_results.keys():
        results[key] = (train_results[key], val_results[key], test_results[key])
    return results
