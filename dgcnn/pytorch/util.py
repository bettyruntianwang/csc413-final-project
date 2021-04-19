#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from itertools import permutations
import math

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def cross_entropy_loss(pred_labels, true_labels):
    # pred labels: batch size x num pts x pt num
    # true labels: batch size x num pts
    pred_labels = pred_labels.view(pred_labels.shape[0]*pred_labels.shape[1], -1)
    true_labels = true_labels.view(-1)
    per_instance_seg_loss = F.cross_entropy(input=pred_labels, target=true_labels)
    #seg_loss = tf.reduce_mean(per_instance_seg_loss)
    #per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
  
    #return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res
    return per_instance_seg_loss

def cal_min_pairwise_seg_loss(pred_labels, true_labels):
    part_counts = torch.max(true_labels, axis=1)[0] + 1
    batch_num = pred_labels.shape[0]
    total_loss = 0.0
    permuted_labels = -1 * torch.ones_like(true_labels).to(true_labels.device)
    for i in range(batch_num):
        perm_count = math.factorial(part_counts[i]-1)   # don't count the base part into permutation, base part is always 0
        pred_l = pred_labels[i].unsqueeze(dim=0).repeat(perm_count, 1, 1)     
        true_l = true_labels[i].unsqueeze(dim=0).repeat(perm_count, 1)

        part_index_permutations = torch.tensor(list(permutations(range(1, part_counts[i])))).to(true_labels.device)

        for part_value in range(1, part_counts[i]):
            label_mask = (true_labels[i] == part_value).nonzero().view(-1)
            true_l[:, label_mask] = part_index_permutations[:, part_value-1].unsqueeze(dim=1).repeat(1, label_mask.shape[0])

        ce_loss = F.cross_entropy(input=pred_l.view(pred_l.shape[0]*pred_l.shape[1], pred_l.shape[2]), 
                                target=true_l.view(true_l.shape[0]*true_l.shape[1]), reduction='none').view(perm_count, -1)

        all_cs_loss = torch.mean(ce_loss, dim=1)
        per_object_min_ce_loss = torch.min(all_cs_loss)
        min_cs_loss_index = torch.argmin(all_cs_loss)
        permuted_labels[i] = true_l[min_cs_loss_index]

        total_loss += per_object_min_ce_loss

    return total_loss / batch_num, permuted_labels

def compute_volumn_from_bbox(point_min, point_max):
    return torch.prod(torch.abs(point_max - point_min))

def cal_iou_loss(pred_points, true_points):
    if pred_points.shape[0] == 0 or true_points.shape[0] == 0:
        return torch.tensor(0.0000001, dtype=torch.float32, requires_grad=True) # give very small number to avoid inf after inverting

    # calculate iou loss for two point clouds
    pred_min = torch.min(pred_points, dim=0)[0]
    pred_max = torch.max(pred_points, dim=0)[0]
    true_min = torch.min(true_points, dim=0)[0]
    true_max = torch.max(true_points, dim=0)[0]

    intersect_min = torch.max(pred_min, true_min)
    intersect_max = torch.min(pred_max, true_max)

    intersect_volume = compute_volumn_from_bbox(intersect_min, intersect_max)
    union_volume = compute_volumn_from_bbox(pred_min, pred_max) + compute_volumn_from_bbox(true_min, true_max) - intersect_volume

    return intersect_volume / union_volume

def cal_max_part_iou_loss(part_point_list1, part_point_list2, self_compare=False):
    max_part_iou = torch.zeros(len(part_point_list1), requires_grad=False) # in-place modification shouldn't require gradient
    for i, part_points in enumerate(part_point_list1):
        for j, gt_part_points in enumerate(part_point_list2):
            if self_compare and i==j:
                continue
            part_iou = cal_iou_loss(part_points, gt_part_points)

            if part_iou > max_part_iou[i]:
                max_part_iou[i] = part_iou

    return max_part_iou

def get_part_point_cloud_from_label(points, part_labels, part_count):
    # separate a point cloud wrt part labels
    part_points = []
    for p in range(part_count):
        part_point_indices = (part_labels == p).nonzero().view(-1)
        part_points.append(points[part_point_indices])
    
    return part_points

def cal_partwise_iou_los(data, pred, true_labels, part_counts):
    pred_labels = torch.max(pred, dim=2)[1]
    total_part_iou_loss_leaf = torch.tensor(0, dtype=torch.float32, requires_grad=True)
    total_part_iou_loss = total_part_iou_loss_leaf.clone()
    for i, points in enumerate(data):
        part_points = get_part_point_cloud_from_label(points, pred_labels[i], part_counts[i])
        gt_part_points = get_part_point_cloud_from_label(points, true_labels[i], part_counts[i])

        max_gt_part_iou = torch.mean(cal_max_part_iou_loss(part_points, gt_part_points))
        max_pred_part_iou = torch.mean(cal_max_part_iou_loss(part_points, part_points, self_compare=True))

        total_part_iou_loss = torch.add(total_part_iou_loss, 1 / max_gt_part_iou + max_pred_part_iou)

    total_part_iou_loss = torch.div(total_part_iou_loss, data.shape[0])

    return total_part_iou_loss

def cal_partwise_mask_loss(pred_labels, true_labels, part_counts, compute_new_true_labels=False):
    for pred_label, i in enumerate(pred_labels):
        part_count = part_counts[i]
        for part_i in part_count:
            part_i_mask = (torch.max(pred_label, dim=2)[1] == part_i).nonzero()

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


#a = torch.tensor([[0,0,0], [0,1,1], [1,1,1]], dtype=torch.float)
#b = torch.tensor([[0,0,0], [0,2,2], [2,2,2]], dtype=torch.float)

#print(cal_iou_loss(a, b))