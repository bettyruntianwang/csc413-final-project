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

# def pairwise_distance(point_cloud):
#   """Compute pairwise distance of a point cloud.

#   Args:
#     point_cloud: tensor (batch_size, num_points, num_dims)

#   Returns:
#     pairwise distance: (batch_size, num_points, num_points)
#   """
#   print(f'point cloud size before pwd: {point_cloud.size()}')
#   og_batch_size = point_cloud.size(0)
#   point_cloud = torch.squeeze(point_cloud)
#   if og_batch_size == 1:
#     point_cloud = torch.unsqueeze(point_cloud, 0)
    
#   point_cloud_transpose = torch.transpose(point_cloud, 2, 1)
#   point_cloud_inner = -2 * torch.matmul(point_cloud, point_cloud_transpose)
#   point_cloud_square = torch.sum(point_cloud**2, dim=-1, keepdim=True)
#   point_cloud_square_tranpose = torch.transpose(point_cloud_square, 2, 1)
#   return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_edge_feature(x, k=20, nn_idx=None):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  # og_batch_size = point_cloud.size(0)
  # point_cloud = torch.squeeze(point_cloud)

  # if og_batch_size == 1:
  #   point_cloud = torch.unsqueeze(point_cloud, 0)

  # point_cloud_central = point_cloud

  # point_cloud_shape = point_cloud.size()
  # batch_size = point_cloud_shape[0]
  # num_points = point_cloud_shape[1]
  # num_dims = point_cloud_shape[2]

  # idx_ = torch.range(0, batch_size-1) * num_points
  # idx_ = torch.reshape(idx_, [batch_size, 1, 1]) 

  # point_cloud_flat = torch.reshape(point_cloud, [batch_size*num_points, num_dims])
  # indices = (nn_idx+idx_).long()
  # print(f'point cloud flat size: {point_cloud_flat.size()}')
  # print(indices)
  # print(f'indices size: {indices.size()}')
  # point_cloud_neighbors = torch.gather(point_cloud_flat, 0, indices)
  # point_cloud_central = torch.expand_dims(point_cloud_central, dim=-2)

  # point_cloud_central = torch.tile(point_cloud_central, (1, 1, k, 1))
  # print(f'size of pt cloud central {point_cloud_central.size()}')

  # edge_feature = torch.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  # print(f'size of pt edge feature {edge_feature.size()}')

  batch_size = x.size(0)
  num_points = x.size(2)
  x = x.view(batch_size, -1, num_points)
  #device = torch.device('cuda')

  idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

  nn_idx = nn_idx + idx_base

  nn_idx = nn_idx.view(-1)

  _, num_dims, _ = x.size()

  x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
  feature = x.view(batch_size*num_points, -1)[nn_idx, :]
  feature = feature.view(batch_size, num_points, k, num_dims) 
  x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
  
  feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

  return feature

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
    def __init__(self, args, part_num, input_dim, num_points, batch_size):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = 20
        self.part_num = part_num
    
        #TODO: add bn_decay to args
        self.bn_decay = args.bn_decay

        #TODO: batchnorm decay and weight decay
        # momentum default = 0.1
        self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_decay, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(128, momentum=self.bn_decay, eps=1e-3)
        self.bn3 = nn.BatchNorm2d(256, momentum=self.bn_decay, eps=1e-3)
        self.bn5 = nn.BatchNorm2d(1024, momentum=self.bn_decay, eps=1e-3)

        self.bn_fc = nn.BatchNorm1d(256, momentum=self.bn_decay, eps=1e-3)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=self.bn_decay, eps=1e-3)

        self.transform_conv1 = nn.Sequential(nn.Conv2d(input_dim*2, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU())

        self.transform_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=True),
                                   self.bn2,
                                   nn.ReLU())
        
        self.transform_conv3 = nn.Sequential(nn.Conv2d(128, 1024, kernel_size=1, bias=True),
                                   self.bn5,
                                   nn.ReLU())


        #TODO: check tf_util: fully connected is not nn.linear
        #TODO: check this dimension
        self.transform_fc1 = nn.Sequential(nn.Linear(1024*10, 512, bias=True),
                                            self.bn_fc2,
                                            nn.ReLU())

        self.transform_fc2 = nn.Sequential(nn.Linear(512, 256, bias=True),
                                            self.bn_fc,
                                            nn.ReLU())

        self.transform_fc3 = nn.Linear(256, 9, bias=True)

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Conv2d(64*3, 64, kernel_size=1, bias=True),
                                   self.bn1,
                                   nn.ReLU(),
                                   nn.MaxPool2d((num_points,1), stride=2))

        self.conv7 = nn.Sequential(nn.Conv2d(64*4, 256, kernel_size=1, bias=True),
                                   self.bn3,
                                   nn.ReLU(),
                                   nn.Dropout(p=0.4)
                                   )

        self.conv8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=True),
                                   self.bn3,
                                   nn.ReLU(),
                                   nn.Dropout(p=0.4)
                                   )

        self.conv9 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=True),
                                   self.bn2,
                                   nn.ReLU())
        
        self.conv10 = nn.Sequential(nn.Conv2d(128, part_num, kernel_size=1, bias=True),
                                   nn.ReLU())


    def input_transform_net(self, edge_feature, bn_decay=0.1, K=3, is_dist=False):
        net = self.transform_conv1(edge_feature)
        net = self.transform_conv2(net)
        #net = torch.max(net, dim=-2, keepdim=True)[0]
        net = self.transform_conv3(net)

        batch_size, double_num_dims, num_points, num_neighb = edge_feature.size()
        net = F.max_pool2d(input=net, kernel_size=(num_points, 1), stride=2)
        net = net.reshape([batch_size, -1])

        net = self.transform_fc1(net)
        net = self.transform_fc2(net)

        # TODO: check the added bias
        net = self.transform_fc3(net)
        net += torch.tensor(np.eye(K).flatten()).to(net.device)

        transform = torch.reshape(net, (batch_size, K, K))
        return transform

    def forward(self, x):
        # x: batch_size x 3 x num_point
        batch_size = x.size(0)
        num_point = x.size(2)

        input_image = torch.unsqueeze(x, -1)
        
        nn_idx = knn(x, k=self.k)
        
        edge_feature = get_edge_feature(x, k=self.k, nn_idx=nn_idx)
        transform = self.input_transform_net(edge_feature)
        point_cloud_transformed = torch.matmul(x.permute(0, 2, 1), transform)

        input_image = point_cloud_transformed.permute(0, 2, 1)

        nn_idx = knn(input_image, k=self.k)
        edge_feature = get_edge_feature(input_image, k=self.k, nn_idx=nn_idx)   # batch_size x channel (num_dim) x num_pt x k

        x = self.conv1(edge_feature)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # batch_size x channel1 x num_pt

        nn_idx = knn(x1, k=self.k)
        x = get_edge_feature(x1, k=self.k, nn_idx=nn_idx)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0] 

        nn_idx = knn(x2, k=self.k)
        x = get_edge_feature(x2, k=self.k, nn_idx=nn_idx)
        x = self.conv5(x) # batch size x channel size x # points x k
        x3 = x.max(dim=-1, keepdim=False)[0] # batch size x channel size x # points

        catted_xs = torch.cat((x1, x2, x3), dim=1).unsqueeze(-1) # batch size x channel size x # pts x 1
        out_max = self.conv6(catted_xs) # batch size x channel size x 1 x 1
        # removed one_hot_label_expand from part_seg_model

        expand = out_max.repeat(1, 1, num_point, 1) # batch size x channel size x #pts x 1
        concat = torch.cat((expand, catted_xs), dim=1)
    
        result = self.conv7(concat) #includes dropout
        result = self.conv8(result)
        result = self.conv9(result)
        result = self.conv10(result) # batch_size x part_num x num_pt x 1

        result = result.squeeze(-1).permute(0, 2, 1)    # batch_size x num_pt x part_num

        return result
