import scipy
import numpy as np
from scipy.linalg import pinv
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
from torch_geometric.utils import degree, coalesce, remove_self_loops, to_undirected

def largest_singular_value(A, k=1):
    '''
    Return the largest singular value of A
    '''
    _, D, _ = svds(A, k=k)
    return D[::-1]

def largest_3_singular_values(A):
    '''
    A is a scipy sparse matrix
    return the largest 5 eigenvalues of A
    '''
    _, D, _ = svds(A, k=3)
    return D[::-1]

def to_edge_vertex_matrix(edge_index, num_nodes):
    '''
    return edge-vertex incidence matrix ExV
    '''
    rows, cols = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    num_edges = edge_index.size(1)
    B = sp.coo_matrix(
        (([1] * num_edges) + ([-1] * num_edges),
        (list(range(num_edges)) * 2, list(rows) + list(cols))), 
        shape=[num_edges, num_nodes]
    )    
    return B

def to_edge_index(edge_vertex_matrix, edge_weights):
    '''
    Transform B, w to edge_index and edge_weights as Tensors

    return edge_index, edge_weights (Both are tensors passed into GNNs)
    '''
    edge_index = []
    for i in edge_vertex_matrix: # get rid of this for loop
        head = np.where(i == 1)[0][0]
        tail = np.where(i == -1)[0][0]
        edge_index.append([head, tail])
    edge_index = np.array(edge_index).T
    edge_index = torch.from_numpy(edge_index)
    edge_weights = torch.from_numpy(edge_weights).float()
    return edge_index, edge_weights

def uniform_sampling(B, w, p):
    '''
    B: edge-vertex incidence matrix
    w: edge weights
    p: remaining ratio, i.e., (1-p) is the removal ratio
    '''
    m = B.shape[0]
    sampled_edges = np.random.binomial(1, p, size=m)
    sampled_edges = np.expand_dims(sampled_edges, 1)

    tilde_B = B * sampled_edges
    tilde_w = 1/p * w * sampled_edges[:, 0]
    return tilde_B, tilde_w

def compute_graphsaint_scores(edge_index, num_nodes, num_sample_edges):
    '''
    compute the probability of each edge to be sampled with expectation of num_sampled_edges
    '''
    degrees = degree(edge_index[0], num_nodes)
    degrees = degrees.cpu().numpy()
    
    edge_index = edge_index.cpu().numpy()
    probs = 1/degrees[edge_index[0]] + 1/degrees[edge_index[1]]
    probs = probs/probs.sum()
    probs = probs*num_sample_edges
    probs[probs>1] = 1
    return probs

def graphsaint_edge_sampling(w, probs, edge_index):
    '''
    probs: a array of probabilities for each edge
    '''
    sampled_edges = np.random.binomial(1, probs) # 1 means sampled, 0 means not sampled
    sampled_edge_index = edge_index[:, sampled_edges==1]
    sampled_edges = np.expand_dims(sampled_edges, 1)
    tilde_w = 1/probs * w * sampled_edges[:, 0]
    # tilde_w = tilde_w[sampled_edges[:, 0] == 1]
    return sampled_edge_index, tilde_w

'''
Helper functions from https://github.com/WodkaRHR/graph_sparsification_by_effective_resistances 
'''
def compute_Z(L, W_sqrt, B, k=100, eta=1e-3, max_iters=1000, convergence_after = 100,
                                    tolerance=1e-2, log_every=10, compute_exact_loss=False, edge_index=None, batch_size_Z = 128, batch_size_L = 1024):
    """ Computes the Z matrix using gradient descent.
    
    Parameters:
    -----------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    eta : float
        Step size for the gradient descent.
    max_iters : int
        Maximum number of iterations.
    convergence_after : int
        If the loss did not decrease significantly for this amount of iterations, the gradient descent will abort.
    tolerance : float
        The minimum amount of energy decrease that is expected for iterations. If for a certain number of iterations
        no overall energy decrease is detected, the gradient descent will abort.
    log_every : int
        Log the loss after each log_every iterations.
    compute_exact_loss : bool
        Only for debugging. If set it computes the actual pseudo inverse without down-projection and checks if
        the pairwise distances in Z's columns are the same with respect to the forbenius norm.
        
    Returns:
    --------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate resistances.
    """
    # Compute the random projection matrix
    # Theoretical value of k := int(np.ceil(np.log(B.shape[1] / epsilon**2))), However the constant could be large.
    Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
    Q *= 1 / np.sqrt(k)
    Y = (W_sqrt @ B).tocsr()
    Y_red = Q @ Y

    if compute_exact_loss:
        # Use exact effective resistances to track actual similarity of the pairwise distances
        L_inv = np.linalg.pinv(L.todense())
        Z_gnd = sp.csr_matrix.dot(Y, L_inv)
        pairwise_dist_gnd = Z_gnd.T.dot(Z_gnd)
    
    # Use gradient descent to solve for Z
    Z = np.random.randn(k, L.shape[1])/(2*np.math.sqrt(k))
    best_loss = np.inf
    best_iter = np.inf

    best_Z = None

    for it in range(max_iters):
        batch_Z = np.random.choice(k, batch_size_Z, replace=False)
        batch_L = np.random.choice(L.shape[1], batch_size_L, replace=False)
        
        residual = Y_red[batch_Z][:, batch_L] - Z[batch_Z, :] @ L[:, batch_L]
        loss = np.linalg.norm(residual)
        if it % log_every == 0: 
            leverage_scores = compute_effective_resistances(Z, edge_index.numpy())
            print(f'Loss before iteration {it}: {loss}')
            print(f'Leverage score before iterations: {it}: {leverage_scores.sum()}')
            if compute_exact_loss:
                pairwise_dist = Z.T.dot(Z)
                exact_loss = np.linalg.norm(pairwise_dist - pairwise_dist_gnd)
                print(f'Loss w.r.t. exact pairwise distances {exact_loss}')
        
        if loss + tolerance < best_loss:
            best_loss = loss
            best_iter = it
            best_Z = Z.copy()
        elif it > best_iter + convergence_after:
            # No improvement for 100 iterations
            print(f'Convergence after {it - 1} iterations.')
            break
        
        Z[batch_Z] += eta * residual @ (L[:, batch_L]).T # L.dot(residual.T).T
    print(f"Best loss: {best_loss}")
    return best_Z

def spsolve_Z(L, W_sqrt, B, k=100):
    """ Computes the Z matrix using gradient descent.
    
    Parameters:
    -----------
    L : sp.csr_matrix, shape [N, N]
        The graph laplacian, unnormalized.
    W_sqrt : sp.coo_matrix, shape [e, e]
        Diagonal matrix containing the square root of weights of each edge.
    B : sp.coo_matrix, shape [e, N]
        Signed vertex incidence matrix.
    
    Returns:
    --------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate resistances.
    """
    # Compute the random projection matrix
    # Theoretical value of k := int(np.ceil(np.log(B.shape[1] / epsilon**2))), However the constant could be large.
    Q = (2 * np.random.randint(2, size=(k, B.shape[0])) - 1).astype(np.float)
    Q *= 1 / np.sqrt(k)
    Y = (W_sqrt @ B).tocsr()
    Y_red = Q @ Y
    L_T = L.T
    
    Z = []
    for i in range(k):
        Z_i = sp.linalg.minres(L_T, Y_red[i, :])[0]
        Z.append(Z_i)
    Z = np.stack(Z, axis=0)
    return Z

def compute_effective_resistances(Z, edges):
    """ Computes the effective resistance for each edge in the graph.
    
    Paramters:
    ----------
    Z : ndarray, shape [k, N]
        Matrix from which to efficiently compute approximate effective resistances.
    edges : tuple
        A tuple of lists indicating the row and column indices of edges.
        
    Returns:
    --------
    R : ndarray, shape [e]
        Effective resistances for each edge.
    """
    rows, cols = edges
    assert(len(rows) == len(cols))
    R = []
    # Compute pairwise distances
    for i, j in zip(rows, cols):
        R.append(np.linalg.norm(Z[:, i] - Z[:, j]) ** 2)
    return np.array(R)

def approximate_leverage_score(B, L, W_sqrt, edge_index, k, Z_dir = None):
    if Z_dir: # Load Z if it exists
        Z = np.load(Z_dir)
    else:
        Z = spsolve_Z(L, W_sqrt, B, k=k)
        # Z = compute_Z(L, W_sqrt, B, k=k, max_iters=max_iters, log_every=log_every, eta=eta,
        #              edge_index=edge_index, batch_size_Z = batch_size_Z, batch_size_L=batch_size_L) # We need to tune k; My experience is k = 20*nlogn
    # store Z in a file
    # np.save('../save_z/dataset_name.npy', Z)
    scores = compute_effective_resistances(Z, edge_index.numpy())
    return scores

def compute_leverage_score(B, L):
    '''
    A is a numpy array
    returns a numpy array of leverage scores
    '''
    scores = B @ pinv(L) @ B.T
    scores = np.diagonal(scores)
    return scores
    
def compute_laplacian(B, W):
    '''
    B is a numpy array
    W is a numpy array
    returns the laplacian matrix
    '''
    W = np.squeeze(np.asarray(W))
    w = sp.diags(W)
    return B.T @ w @ B

def leverage_score_sampling(B, w, alpha, approx_scores=False, edge_index=None, k=200, Z_dir = None):
    '''
    B: edge-vertex incidence matrix
    w: edge weights
    alpha: a parameter that controls the sampling rate. if alpha = 1, sum of probability should be nlogn 
    
    Return a sampled graph
    '''
    L = B.T @ sp.diags(w) @ B
    if approx_scores:
        W_sqrt = sp.diags(np.sqrt(w))
        assert edge_index is not None
        scores = approximate_leverage_score(B, L, W_sqrt, edge_index=edge_index, k=k, Z_dir = Z_dir)
    else:
        L = L.todense()
        scores = compute_leverage_score(B, L) # b^T (L)^{-1} b

    sample_prob = alpha*w*scores # probability of sampling each edge
    sample_prob[sample_prob>1] = 1.0

    sampled_edges = np.random.binomial(1, sample_prob)
    sampled_edge_index = edge_index[:, sampled_edges==1] # obtain the sampled edge index
    sampled_edges = np.expand_dims(sampled_edges, 1)

    tilde_B = B.multiply(sampled_edges)
    tilde_w = 1/sample_prob * w * sampled_edges[:, 0]

    return tilde_B, tilde_w, sampled_edge_index

def remove_redundant_edges(data):
    edge_index = data.edge_index
    new_edge_index = coalesce(edge_index, num_nodes=data.num_nodes)
    new_edge_index = remove_self_loops(new_edge_index)[0]
    new_edge_index = new_edge_index[:, new_edge_index[0] < new_edge_index[1]] # if (i,j) is an edge, then remove (j,i) if it exists
    return new_edge_index

def effective_resistance_sample_graph(data, alpha, approx_scores=False, k=200, Z_dir = None): 
    assert alpha > 0
    edge_index = remove_redundant_edges(data)
    B = to_edge_vertex_matrix(edge_index, data.num_nodes)
    w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()
    _, tilde_w, sampled_edge_index = leverage_score_sampling(B, w, alpha, approx_scores=approx_scores, edge_index=edge_index, k=k, Z_dir = Z_dir)

    return sampled_edge_index, torch.from_numpy(tilde_w).float()

def effective_resistance_sample_graph_test(data, alpha, approx_scores=False, k=200, Z_dir = None): 
    assert alpha > 0
    edge_index = remove_redundant_edges(data)
    B = to_edge_vertex_matrix(edge_index, data.num_nodes)
    w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()
    tilde_B, tilde_w, sampled_edge_index = leverage_score_sampling(B, w, alpha, approx_scores=approx_scores, edge_index=edge_index, k=k, Z_dir = Z_dir)

    # sampled_edge_index, tilde_w = to_undirected(sampled_edge_index, torch.from_numpy(tilde_w).float(), num_nodes=data.num_nodes)
    # tilde_B = to_edge_vertex_matrix(sampled_edge_index, data.num_nodes)
    tilde_L = compute_laplacian(tilde_B, tilde_w)

    return tilde_B, tilde_w, sampled_edge_index, largest_3_singular_values(tilde_L)

def graphsaint_sample_graph(data, sampled_edges):
    edge_index = remove_redundant_edges(data)
    w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()
    probs = compute_graphsaint_scores(edge_index, data.num_nodes, sampled_edges)
    sampled_edge_index, tilde_w = graphsaint_edge_sampling(w, probs, edge_index)
    
    tilde_w = tilde_w[tilde_w!=0]

    return sampled_edge_index, torch.from_numpy(tilde_w).float()

def graphsaint_sample_graph_test(data, sampled_edges):
    edge_index = remove_redundant_edges(data)
    w = np.ones(edge_index.shape[1]) if data.edge_weight is None else data.edge_weight.numpy()
    probs = compute_graphsaint_scores(edge_index, data.num_nodes, sampled_edges)
    sampled_edge_index, tilde_w = graphsaint_edge_sampling(w, probs, edge_index)
    
    tilde_w = tilde_w[tilde_w!=0]
    
    sampled_edge_index, tilde_w = to_undirected(sampled_edge_index, torch.from_numpy(tilde_w).float(), num_nodes=data.num_nodes)
    tilde_B = to_edge_vertex_matrix(sampled_edge_index, data.num_nodes)
    tilde_L = compute_laplacian(tilde_B, tilde_w)

    return sampled_edge_index, tilde_w, largest_3_singular_values(tilde_L)
