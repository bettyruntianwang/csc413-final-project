#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.size(0)
  point_cloud = torch.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = torch.unsqueeze(point_cloud, 0)
    
  point_cloud_transpose = torch.transpose(point_cloud, 2, 1)
  point_cloud_inner = -2 * torch.matmul(point_cloud, point_cloud_transpose)
  point_cloud_square = torch.sum(point_cloud**2, dim=-1, keepdim=True)
  point_cloud_square_tranpose = torch.transpose(point_cloud_square, 2, 1)
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(x, k):
    dists = pairwise_distance(x)
    idx = dists.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

# def knn(x, k):
#     inner = -2*torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
#     return idx

def get_edge_feature(point_cloud, k=20, nn_idx):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.size(0)
  point_cloud = torch.squeeze(point_cloud)

  if og_batch_size == 1:
    point_cloud = torch.unsqueeze(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.size()
  batch_size = point_cloud_shape[0]
  num_points = point_cloud_shape[1]
  num_dims = point_cloud_shape[2]

  idx_ = torch.range(batch_size) * num_points
  idx_ = torch.reshape(idx_, [batch_size, 1, 1]) 

  point_cloud_flat = torch.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = torch.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = torch.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = torch.tile(point_cloud_central, (1, 1, k, 1))

  edge_feature = torch.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  return edge_feature

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = 20
        
        #TODO: batchnorm decay and weight decay
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.transform_conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())

        self.transform_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        
        self.transform_conv3 = nn.Sequential(nn.Conv2d(128, 1024, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())

        #TODO: check tf_util: fully connected is not linear
        self.transform_fc1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.transform_fc2 = nn.Linear(512, 256, bias=False)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU()
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
    
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def input_transform_net(edge_feature, is_training, bn_decay=None, K=3, is_dist=False):
        net = self.transform_conv1(edge_feature)
        net = self.transform_conv2(net)
        net = x.max(dim=-2, keepdim=True)[0]
        net = self.transform_conv3(net)

        num_points = edge_feature.size(1)
        net = nn.MaxPool2d((num_points, 1), stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        num_point = x.size(1)

        input_image = torch.unsqueeze(x, -1)
        
        nn_idx = knn(x, k=self.k)
        
        edge_feature = get_edge_feature(x, k=self.k, nn_idx=nn_idx)
        transform

        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-2, keepdim=True)[0]

        x = get_edge_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
