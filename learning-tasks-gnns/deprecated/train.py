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