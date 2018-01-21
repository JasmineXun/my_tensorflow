# encoding:utf-8
import pickle as pkl
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


def get_matrixM(path):
    res = {}
    user_list = []
    item_list = []
    with open(path) as f: # B46_154611_chengji.csv
        line = f.readline()
        while line:
            eles = line.split(',')
            if eles[0] not in res.keys():
                res[eles[0]] = {}
                user_list.append(eles[0])
            if eles[1] not in item_list:
                item_list.append(eles[1])
            try:
                res[eles[0]][eles[1]] = int(float(eles[2])/10)
            except:
                pass
            line = f.readline()

    data=pd.DataFrame(res).T # 成绩矩阵，行表示用户，列表示课程
    data=pd.DataFrame.fillna(data,0)# 用０填充nan

    c=np.array(data.columns).reshape([data.shape[1],1],)
    r=np.array(data.index).reshape([data.shape[0],1])
    r_c=np.vstack((r,c))
    # np.savetxt('./data/meta.txt', r_c)
    print('r_c={0}'.format(r_c))

    M, N = data.shape
    '''
    data=np.array(data)
    matrix_all=[] # 存储１－１０个级别对应的0/1矩阵
    for i in range(10):
        matrix=np.zeros([M,N])
        matrix_all.append(matrix)
    for row in range(M):
        for cow in range(N):
            if not np.isnan(data[row][cow]):
                matrix_all[int(data[row][cow]-1)][row][cow]=1  #在对应值的对应级别上修改矩阵
    '''
    one_hot_data = np.array([one for one in range(N+M)])  #
    data_onehot = pd.get_dummies(one_hot_data)
    return data_onehot,data

def get_adj_01(adj):
    M, N = adj.shape
    matrix_all = []  # 存储１－１０个级别对应的0/1矩阵
    for i in range(10):
        matrix = np.zeros([M, N])
        matrix_all.append(matrix)
    for row in range(M):
        for cow in range(N):
            if int(adj[row][cow]) != 0:
            #if not np.isnan(adj[row][cow]):
                matrix_all[int(adj[row][cow] - 1)][row][cow] = 1  # 在对应值的对应级别上修改矩阵

    return matrix_all

def get_marix_conbine_matixT(matrix):# 矩阵matrix和其转置矩阵分在处在对角位置组成对称的邻接矩阵
    matrixT = matrix.T
    zero_h = np.zeros([matrix.shape[0], matrix.shape[0]])
    print (zero_h)
    matrix = np.hstack((zero_h, matrix))
    zero_v = np.zeros([matrixT.shape[0], matrixT.shape[0]])
    matrixT = np.hstack((matrixT, zero_v))
    new = np.vstack((matrix, matrixT))
    return new

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    # print ('graph:{0}'.format(graph)) {0: [633, 1862, 2582], 1: [2, 652, 654], 2: [1986, 332, 1666, 1, 1454], 3: [2544], 4: ...

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def getGraph(path):
    res = {}
    user_list = []
    item_list = []
    with open(path) as f:
        line = f.readline()
        while line:
            eles = line.split(',')
            if eles[0] not in res.keys():
                res[eles[0]] = {}
                user_list.append(eles[0])
            if eles[1] not in item_list:
                item_list.append(eles[1])
            res[eles[0]][eles[1]] = float(eles[2])
            line = f.readline()

    N = len(user_list) + len(item_list)
    print(len(user_list))
    print(len(item_list))
    i = 0
    id_map = dict()
    for user in user_list:
        id_map[user] = i
        i = i + 1
    for item in item_list:
        id_map[item] = i
        i = i + 1

    adj = np.zeros([N, N])
    for key_user in res:
        for key_item in res[key_user]:
            adj[id_map[key_user ]][id_map[key_item ]]=res[key_user][key_item]
            adj[id_map[key_item ]][id_map[key_user]] = res[key_user][key_item]

    one_hot_data=np.array([one for one in range(N)])  #
    data_onehot=pd.get_dummies(one_hot_data)
    return adj,data_onehot
def getFeature():
    # student_kch_feature_p_350.txt student_kch_feature_q_350.txt
    feature_user = pd.read_csv('./data/student_kch_feature_p_350.txt', sep=',', header=None)
    feature_item = pd.read_csv('./data/student_kch_feature_q_350.txt', sep=',', header=None)

    feature_item = feature_item.T
    feature_x = np.vstack((feature_user, feature_item))
    print('user shape:{0},item shape:{1}'.format(feature_user.shape, feature_item.shape))
    print('feature_x shape:{0}'.format(feature_x.shape))
    return feature_x
if __name__=='__main__':
    print()
    # data, adj = getData('./data/B46_154611_chengji.csv')
    #adj = sp.coo_matrix(adj)
    #print(adj.shape)

