import numpy as np
import torch
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array, check_symmetric
from scipy.linalg import orth
import scipy.sparse as sparse
from munkres import Munkres


def regularizer_pnorm(c, p):
    return torch.pow(torch.abs(c), p).sum()


def sklearn_predict(A, n_clusters):
    spec = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    res = spec.fit_predict(A)
    return res


def accuracy(pred, labels):
    err = err_rate(labels, pred)                
    acc = 1 - err
    return acc


def subspace_preserving_error(A, labels, n_clusters):
    one_hot_labels = torch.zeros([A.shape[0], n_clusters])
    for i in range(A.shape[0]):
        one_hot_labels[i][labels[i]] = 1.0
    mask = one_hot_labels.matmul(one_hot_labels.T)
    l1_norm = torch.norm(A, p=1, dim=1)
    masked_l1_norm = torch.norm(mask * A, p=1, dim=1)
    e = torch.mean((1. - masked_l1_norm / l1_norm)) * 100.
    return e


def normalized_laplacian(A):
    D = torch.sum(A, dim=1)
    D_sqrt = torch.diag(1.0 / torch.sqrt(D))
    L = torch.eye(A.shape[0]) - D_sqrt.matmul(A).matmul(D_sqrt)
    return L
    
    
def connectivity(A, labels, n_clusters):
    c = []
    for i in range(n_clusters):
        A_i = A[labels == i][:, labels == i]
        L_i = normalized_laplacian(A_i)
        eig_vals, _ = torch.symeig(L_i)
        c.append(eig_vals[1])
    return np.min(c)


def topK(A, k, sym=True):
    """ 
    Return a new matrix with only row-wise top k entries preserved.
    
    Args:
        A:
        k:
        sym: 
    Returns:
    """
    val, indicies = torch.topk(A, dim=1, k=k)
    Coef = torch.zeros_like(A).scatter_(1, indicies, val)
    if sym:
        Coef = (Coef + Coef.t()) / 2.0
    return Coef


def best_map(L1, L2):
    """ 
    Rearrange the cluster label to minimize the error rate using the Kuhn-Munkres algorithm.
    Fetched from https://github.com/panji1990/Deep-subspace-clustering-networks

    Args:
        L1 (list): ground truth label.
        L2 (list): clustering result.
    Return:
        (list): rearranged predicted result.
    """
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    """
    Get error rate of the cluster result.
    Fetched from https://github.com/panji1990/Deep-subspace-clustering-networks
    Args:
        gt_s (list): ground truth label.
        s (list): clustering result.
    Return:
        (float): clustering error.
    """
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate 


def gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, noise_level=0.0):
    """This funtion generates a union of subspaces under random model, i.e., 
    subspaces are independently and uniformly distributed in the ambient space,
    data points are independently and uniformly distributed on the unit sphere of each subspace

    Parameters
    -----------
    ambient_dim : int
        Dimention of the ambient space
    subspace_dim : int
        Dimension of each subspace (all subspaces have the same dimension)
    num_subspaces : int
        Number of subspaces to be generated
    num_points_per_subspace : int
        Number of data points from each of the subspaces
    noise_level : float
        Amount of Gaussian noise on data
		
    Returns
    -------
    data : shape (num_subspaces * num_points_per_subspace) by ambient_dim
        Data matrix containing points drawn from a union of subspaces as its rows
    label : shape (num_subspaces * num_points_per_subspace)
        Membership of each data point to the subspace it lies in
    """

    data = np.empty((num_points_per_subspace* num_subspaces, ambient_dim))
    label = np.empty(num_points_per_subspace * num_subspaces, dtype=int)
  
    for i in range(num_subspaces):
        basis = np.random.normal(size=(ambient_dim, subspace_dim))
        basis = orth(basis)
        coeff = np.random.normal(size=(subspace_dim, num_points_per_subspace))
        coeff = normalize(coeff, norm='l2', axis=0, copy=False)
        data_per_subspace = np.matmul(basis, coeff).T

        base_index = i*num_points_per_subspace
        data[(0+base_index):(num_points_per_subspace+base_index), :] = data_per_subspace
        label[0+base_index:num_points_per_subspace+base_index,] = i

    data += np.random.normal(size=(num_points_per_subspace * num_subspaces, ambient_dim)) * noise_level
  
    return data, label


def dim_reduction(X, dim):
    """Dimension reduction by principal component analysis
		Let X^T = U S V^T be the SVD of X^T in which the singular values are
	in ascending order. The output Xp^T is the last `dim` rows of S * V^T.
  
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
    dim: int
        Target dimension. 
		
    Returns
    -------
    Xp : shape (n_samples, dim)
        Dimension reduced data
	"""
    if dim == 0:
        return X

    w, v = np.linalg.eigh(X.T @ X)
  
    return X @ v[:, -dim:]


def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)


def minmax_normalize(x, p=2):
    rmax, _ = torch.max(x, dim=1, keepdim=True)
    rmin, _ = torch.min(x, dim=1, keepdim=True)
    x = (x - rmin) / (rmax - rmin)
    return x


def spectral_clustering(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    random_state = check_random_state(seed)

    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, 
                                 k=k, sigma=None, which='LA')
    embedding = normalize(vec)
    _, labels_, _ = cluster.k_means(embedding, n_clusters, 
                                         random_state=seed, n_init=n_init)
    return labels_