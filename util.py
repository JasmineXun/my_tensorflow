#-*- coding:utf-8 -*-

"""
Author:xunying/Jasmine
Data:18-1-15
Time:下午4:26
"""
from sklearn.decomposition import ProjectedGradientNMF
import numpy as np
import scipy.sparse as sp

from input_data import get_matrixM
# from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
def baseline_NMF(M):
    # A = np.random.uniform(size=[40, 30])
    nmf_model = ProjectedGradientNMF(n_components=5, init='random', random_state=0)
    nmf_model.fit(M)
    W = nmf_model.fit_transform(M)
    H = nmf_model.components_
    print('shapeW={0};shapeH={1}'.format(W.shape,H.shape))
    # print(np.dot(W,H))
    return np.dot(W,H)
    #np.dot(W,H)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = [adj.toarray()[i[0], i[1]] for i in train_edges]


    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    # return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    return adj_train, train_edges, val_edges, test_edges

def get_roc_score_r(edges_pos, adj_M,adj_M_hat,emb=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    #adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        # preds.append(adj_rec[e[0], e[1]])
        # pos.append(adj_orig[e[0], e[1]])
        preds.append(adj_M_hat[e[0], e[1]])
        pos.append(adj_M[e[0], e[1]])
    loss_mean = np.sqrt(np.mean((np.array(pos)-np.array(preds))**2))
    print('loss_mean:{0}'.format(loss_mean))

    return loss_mean

data_one_hot,data=get_matrixM('./data/B46_154611_chengji.csv')

adj_M_org = np.array(data)
adj=sp.csr_matrix(data)

adj_train, train_edges, val_edges, test_edges=mask_test_edges(adj)
adj_M_hat = baseline_NMF(adj_train)
print('shapeW={0};shapeH={1}'.format(adj_M_org.shape,adj_M_hat.shape))
loss_mse=get_roc_score_r(test_edges,adj_M_org,adj_M_hat)
print(loss_mse)
# a=np.array([[1,2,3],[1,2,2]])
# b=np.array([[2,3,4]])
# print(np.dot(a,b.T))