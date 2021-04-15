import tensorflow as tf
import numpy as np
import math
import os
import sys
from itertools import permutations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util
from transform_nets import input_transform_net

def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 20

  adj = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3, is_dist=True)
  point_cloud_transformed = tf.matmul(point_cloud, transform)
  
  input_image = tf.expand_dims(point_cloud_transformed, -1)
  adj = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)



  adj = tf_util.pairwise_distance(net_1)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util.conv2d(out3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
  
  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
  
  

  adj = tf_util.pairwise_distance(net_2)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  # out6 = tf_util.conv2d(out5, 64, [1,1],
  #                      padding='VALID', stride=[1,1],
  #                      bn=True, is_training=is_training, weight_decay=weight_decay,
  #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

  net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)



  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')


  one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
  one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
  out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     net_1,
                                     net_2,
                                     net_3])

  net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
  net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
  net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
  net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

  net2 = tf.reshape(net2, [batch_size, num_point, part_num])

  return net2


def get_loss(seg_pred, seg):
  per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
  
  return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

# Tianxu: Use this loss function to "improve performance", finger crossed
def get_permutation_invariant_loss(pred_labels, true_labels):
  # pred_labels: batch_size x point_num x part_num
  # true_labels: batch_size x point_num

  sess = tf.InteractiveSession()

  min_parts = tf.reduce_min(true_labels, axis=1, keepdims=True)
  true_labels = true_labels - min_parts # batch_size x point_num
  part_counts = tf.reduce_max(true_labels, axis=1) + 1 # batch_size
  batch_num = pred_labels.shape[0]
  point_num = pred_labels.shape[1]

  per_instance_seg_loss = []
  permuted_labels = []

  for i in range(batch_num):
    perm_base = sess.run(part_counts[i])
    perm_count = math.factorial(perm_base)
    pred_l = tf.repeat(tf.expand_dims(pred_labels[i], axis=0), repeats=perm_count, axis=0)  # perm_num x point_num x part_num
    true_l = np.zeros([perm_count, point_num])            
    part_index_permutations = np.array([list(perm) for perm in permutations(range(perm_base))])

    for part_index in range(perm_base):
      np_true_labels = sess.run(true_labels)
      label_mask = tf.reshape(tf.where(np_true_labels[i] == part_index), [-1])
      true_l[:,sess.run(label_mask)] = np.repeat(np.expand_dims(part_index_permutations[:,part_index], axis=1), repeats=sess.run(tf.shape(label_mask))[0], axis=1)
    true_l = tf.convert_to_tensor(true_l, dtype=tf.int64)

    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_l, labels=true_l)
    all_perm_cs_loss = tf.reduce_mean(ce_loss, axis=1)
    per_instance_seg_loss.append(tf.reduce_min(all_perm_cs_loss))
    min_cs_loss_index = tf.argmin(all_perm_cs_loss)
    permuted_labels.append(true_l[min_cs_loss_index])

  per_instance_seg_loss = tf.convert_to_tensor(per_instance_seg_loss)
  permuted_labels = tf.convert_to_tensor(permuted_labels)

  seg_loss = tf.reduce_mean(per_instance_seg_loss)
  #per_instance_seg_pred_res = tf.argmax(pred_labels, 2)
  
  return seg_loss, permuted_labels

# Test loss functions
seg_pred = tf.constant([[[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],     # [1, 1, 0, 2]
                        [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1]],     # [0, 1, 2, 1]
                        [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.05, 0.9, 0.05]]])  # [1, 0, 0, 1]

seg = tf.constant([[1, 1, 0, 2],
                   [4, 6, 5, 6],
                   [3, 4, 4, 3]])

get_permutation_invariant_loss(seg_pred, seg)
# get_loss(seg_pred, seg)