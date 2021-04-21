#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from simple_model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from util import cal_seg_loss, IOStream, get_part_point_cloud_from_label, cal_min_pairwise_seg_loss
import sklearn.metrics as metrics
import math
import h5py
from matplotlib import pyplot as plt
import yaml
import sys
import time
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from visualization.visualize import plot3d_pts_in_camera_plane, plot3d_pts

VALIDATION_PERCENTAGE = 0.2

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def load_pts_files(pts_file):
    pts = []
    with open(pts_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        pt = [float(value.rstrip()) for value in line.split(' ')]
        pts.append(pt)
    
    return torch.tensor(pts)

# Tianxu: return data and seg
def load_h5_data_seg_Sapien(h5_dir, file_names, num_points=2048):
    data = []
    label = []
    step_counts = []
    part_counts = []
    object_ids = []

    for filename in file_names:
        f = h5py.File(os.path.join(h5_dir, filename))
        root = f['gt_points']

        for cam_index in range(len(root)):
            cam_name = F'cam {cam_index}'
            cam = root[cam_name]

            for step_id in range(len(cam)):
                step_name = F'cam {cam_index} step {step_id}'
                step = cam[step_name]

                data.append([])
                label.append([])

                for part_index in range(len(step)):
                    part_name = F'part {part_index}'
                    part = step[part_name][:]
                    if part.shape[0] == 0:
                        # no point in this part, skip
                        continue
                    if len(data[-1]) == 0:
                        data[-1] = part
                        label[-1] = np.array([part_index] * len(part))
                    else:
                        data[-1] = np.concatenate([data[-1], part], axis=0)
                        label[-1] = np.concatenate([label[-1], np.ones([len(part)]) * part_index], axis=0)

                idx = np.arange(len(label[-1]))
                np.random.shuffle(idx)
                data[-1] = data[-1][idx, :][:num_points, :]
                label[-1] = label[-1][idx][:num_points]
                part_counts.append(len(cam[step_name]))
                step_counts.append(len(cam))
                object_ids.append(int(filename.split('.')[0]))

    return (object_ids, torch.tensor(data, dtype=torch.float), torch.tensor(label, dtype=torch.long), 
            torch.tensor(step_counts, dtype=torch.long), torch.tensor(part_counts, dtype=torch.long))

def get_data_indices(data_size, batch_size=1, val_percentage=VALIDATION_PERCENTAGE):
    total_batch_num = int(data_size / batch_size)
    val_batch_num = math.floor(total_batch_num * val_percentage)
    train_batch_num = total_batch_num - val_batch_num

    train_data_end_index = train_batch_num * batch_size - 1

    return int(train_data_end_index)

def get_data_loaders(dataset, batch_size=1, val_percentage=VALIDATION_PERCENTAGE):
    # Load training and validation data.  
    data_size = len(dataset)

    if val_percentage == 1:
        # used in testing stage
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(0, data_size)), drop_last=True)
        return 0, test_loader

    end_training_index = get_data_indices(data_size, batch_size, val_percentage)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(end_training_index)), drop_last=True)
    test_loader  = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(end_training_index+1, data_size)), drop_last=True)

    return train_loader, test_loader

def test(args, io):
    category = 'laptops-similar-frame'
    perm_loss = True
    data_dir = os.path.join(BASE_DIR, '..', 'part_seg', 'Sapien_part_seg', category)
    total_objects = os.listdir(data_dir)
    np.random.shuffle(total_objects)

    object_ids, data, label, step_counts, part_counts = load_h5_data_seg_Sapien(data_dir, total_objects, args.num_points)

    for i, object_id in enumerate(object_ids):
        yml_load_path = os.path.join(BASE_DIR, '..', 'part_seg', 'Sapien_part_seg', F'{category}-render', str(object_id), 'pose-transformation.yml')                                                     
        with open(yml_load_path, 'r') as f:
            yaml_dict = yaml.load(f)
        proj_matrix = np.array(yaml_dict['projMat']).reshape(4, 4).T
        #view_matrix = np.array(yaml_dict['view_matrix_cam_1']).reshape(4, 4)
        #world_to_image_matrix = np.dot(view_matrix, proj_matrix)
        #pass_once(args, data[i], object_id=object_id, step_id=step_ids[i], project_to_plane=True, transform_matrix=proj_matrix)
        pass_once(args, data[i], object_id=object_id, obj_index=i, project_to_plane=True, transform_matrix=proj_matrix)

    dataset = TensorDataset(data, label, step_counts, part_counts)

    _, test_loader = get_data_loaders(dataset, batch_size=args.test_batch_size, val_percentage=1)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    #Try to load models
    model = DGCNN(args, part_number=6).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()

    test_acc = 0.0
    count = 0.0
    test_loss = 0.0
    test_pred = []
    test_true = []
    for data, label, step_counts, part_counts in test_loader:
        data, label = data.to(device), label.to(device)

        batch_size = data.shape[0]
        logits = model(data.permute(0, 2, 1))
        
        if perm_loss:
            loss, permuted_labels = cal_min_pairwise_seg_loss(logits, label)
            label = permuted_labels
        else:
            loss = cal_seg_loss(logits, label)

        preds = logits.max(dim=2)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().view(-1).numpy())
        test_pred.append(preds.detach().cpu().view(-1).numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)

    test_loss = test_loss*1.0/count
    #test_true = np.concatenate(test_true)
    #test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test loss: %.6f'%(test_acc, test_loss)
    io.cprint(outstr)

def pass_once(args, points, object_id=0, obj_index=0, max_part_count=6, project_to_plane=False, transform_matrix=None):
    output_dir = os.path.join(BASE_DIR, '..', 'visualize_output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_object_dir = os.path.join(output_dir, str(object_id))
    if not os.path.exists(output_object_dir):
        os.mkdir(output_object_dir)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    #Try to load models
    model = DGCNN(args, part_number=6).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()

    logits = model(points.unsqueeze(0).permute(0, 2, 1))

    #for step_id in range(step_count):
    preds = logits[0, :, :].max(dim=1)[1]
    part_points = get_part_point_cloud_from_label(points, preds, max_part_count)

    #pts = load_pts_files('/home/tianxu/Desktop/pair-group/Thesis-project/dgcnn/dgcnn/tensorflow/part_seg/PartAnnotation/03642806/points/1a46d6683450f2dd46c0b76a60ee4644.pts')
    if project_to_plane:
        plot3d_pts_in_camera_plane(part_points, transform_matrix,
                                    pts_name=[f'part {i}: {part_points[i].shape[0]} pts' for i in range(max_part_count)], 
                                    title_name=F'object {object_id}',
                                    show_fig=False, save_fig=True,
                                    save_path=output_object_dir, filename=F'obj {obj_index}', s=10)
    else:
        plot3d_pts([part_points], pts_name=[[f'part {i}: {part_points[i].shape[0]} pts' for i in range(max_part_count)]], 
                                    title_name=F'object {object_id}',
                                    sub_name=F'obj {obj_index}', show_fig=False, save_fig=True,
                                    save_path=output_object_dir, s=10)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,#3072!!!!!!!!!!!!!!! or 2048
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--bn_decay', type=float, default=0.1,
                        help='momentum for batch normalization')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.model_path = os.path.join('checkpoints', args.exp_name, 'models', 'model.h5')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    test(args, io)
