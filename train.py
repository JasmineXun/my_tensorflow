# encoding:utf-8
from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix,auc

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data,getGraph,getFeature,get_adj_01,get_matrixM,get_marix_conbine_matixT
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
#adj, features = load_data(dataset_str)
# 1_4611_chengji.csv B46_154611_chengji.csv xsid_kch_cj_xq_A09_2010-2016
data_onehot,adj= get_matrixM('./data/B46_154611_chengji.csv')
adj=get_marix_conbine_matixT(adj)
adj=sp.csr_matrix(adj)

# features=getFeature()# F-feature
features=data_onehot   # one-hot feature
features=sp.lil_matrix(features)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

#adj_orig=get_adj_01(adj)
#lwith tf.name_scope("vae"):
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_Rs = get_adj_01(adj_train.toarray())
adj=adj_train
adj_R=[sp.csr_matrix(adj_one) for adj_one in adj_Rs]

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_R_norm = [preprocess_graph(one_adj) for one_adj in adj_R]

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': [tf.sparse_placeholder(tf.float32) for _ in range(10)],
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':

        n,_,_=model.logits_output.get_shape().as_list()
        tensor_list=[]
        for adj_channel in placeholders['adj']:
            channel_tensor=tf.reshape(tf.sparse_tensor_to_dense(adj_channel, validate_indices=False), [n,n])
            tensor_list.append(channel_tensor)

        adj_tensor=tf.stack(tensor_list,axis=2)

        opt = OptimizerAE(preds=model.logits_output,
                          labels=adj_tensor,
                          pos_weight=pos_weight,
                          norm=norm)


        # opt = OptimizerAE(preds=model.logits_output,
        #                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
        #                                                               validate_indices=False), model.logits_output.get_shape().as_list()),
        #                   pos_weight=pos_weight,
        #                   norm=norm)
    elif model_str == 'gcn_vae':
        n, _, _ = model.logits_output.get_shape().as_list()
        tensor_list = []
        for adj_channel in placeholders['adj']:
            channel_tensor = tf.reshape(tf.sparse_tensor_to_dense(adj_channel, validate_indices=False), [n, n])
            tensor_list.append(channel_tensor)

        adj_tensor = tf.stack(tensor_list, axis=2)

        opt = OptimizerVAE(preds=model.logits_output,
        # opt = OptimizerVAE(preds=model.reconstructions,  # reconstructions
                          labels=adj_tensor,
                          pos_weight=pos_weight,
                           model=model,
                           num_nodes=n,
                          norm=norm)
        # opt = OptimizerVAE(preds=model,
        #                    labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
        #                                                                validate_indices=False), [-1]),
        #                    model=model, num_nodes=num_nodes,
        #                    pos_weight=pos_weight,
        #                    norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

# 给定混淆矩阵M，计算准确率和召回率
def get_precision_recall_score_confusion(M):
    #preds_all = [2, 2, 4, 3, 4]
    # pre_sore=auc(labels_all,preds_all)
    #M = confusion_matrix(labels_all, preds_all)
    #print (M)
    n = len(M)
    precision=[]
    recall=[]
    for i in range(len(M[0])):
        rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
        try:
            precision.append(M[i][i] / float(colsum))
            recall.append(M[i][i] / float(rowsum))
        except ZeroDivisionError:
            precision.append(0)
            recall.append(0)
    return np.mean([i for i in precision if not np.isnan(i)]),np.mean([i for i in recall if not np.isnan(i)])

def get_roc_score_r(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        adj_rec = sess.run(model.reconstructions,feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    #adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        # preds.append(adj_rec[e[0], e[1]])
        # pos.append(adj_orig[e[0], e[1]])
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        # preds_neg.append(adj_rec[e[0], e[1]])
        # neg.append(adj_orig[e[0], e[1]])
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    # labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    labels_all = np.hstack([pos, neg])
    loss_mean = np.sqrt(np.mean((np.floor(np.array(pos))-np.array(preds))**2))  # by feizhihui
    print('loss_mean:{0}'.format(loss_mean))
    print(pos[:10])
    print(preds[:10])


    # roc_score = roc_auc_score(labels_all, preds_all)
    # ap_score = average_precision_score(labels_all, preds_all)

    return loss_mean,loss_mean
    # return roc_score, ap_score


def get_roc_score(edges_pos, edges_neg, emb=None):
    # if emb is None:
    #     feed_dict.update({placeholders['dropout']: 0})
    #     emb = sess.run(model.z_mean, feed_dict=feed_dict)
    #
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # Predict on test set of edges
    # adj_rec = np.dot(emb, emb.T)
    adj_rec=sess.run(model.reconstructions,feed_dict=feed_dict)
    preds = []
    pos = []
    for e in edges_pos:
        if not np.isnan(adj_orig[e[0], e[1]]):
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        if not np.isnan(adj_orig[e[0], e[1]]):
            preds_neg.append(adj_rec[e[0], e[1]])
            neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])

    #labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    #labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))]) # 当前是多分类问题评估
    labels_all = np.hstack([pos, neg])
    ncorrects = sum(preds_all == labels_all)
    print('ncorrects:'.format(ncorrects))
    '''
    ncorrects = sum(predictions == labels)
    accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
    f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
    F1=2*(precision*recall)/(precision+recall)
    '''

    #roc_score = roc_auc_score(labels_all, preds_all)
    #ap_score = average_precision_score(labels_all, preds_all)
    preds_all=[round(one*10) for one in preds_all]
    confusion=confusion_matrix(labels_all,preds_all)

    precision,recall=get_precision_recall_score_confusion(confusion)
    #print('====',con_score)

    #return roc_score, ap_score
    return precision,recall


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
time_list=[]
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary  adj_orig
    feed_dict = construct_feed_dict(adj_R_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    # writer = tf.summary.FileWriter("logdir", sess.graph) tensorboard
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr = get_roc_score_r(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    time_list.append(time.time() - t)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_MSE=", "{:.5f}".format(val_roc_score[-1]),
          # "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

roc_score, ap_score = get_roc_score_r(test_edges, test_edges_false)
adj_student_kch_embeding = sess.run(model.reconstructions,feed_dict=feed_dict)
np.savetxt('./data/A09student_kch_embedding.txt',adj_student_kch_embeding,delimiter=',')
# print('Test ROC score: ' + str(roc_score))
print('Test MSE: ' + str(ap_score))
# print(val_roc_score)


'''
def get_roc_score_r(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        adj_rec = sess.run(model.reconstructions,feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    #adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        # preds.append(adj_rec[e[0], e[1]])
        # pos.append(adj_orig[e[0], e[1]])
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        # preds_neg.append(adj_rec[e[0], e[1]])
        # neg.append(adj_orig[e[0], e[1]])
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    # labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    labels_all = np.hstack([pos, neg])
    loss_mean = np.sqrt(np.mean((np.array(pos)-np.array(preds))**2))
    print('loss_mean:{0}'.format(loss_mean))
    print(pos[:10])
    print(preds[:10])

'''