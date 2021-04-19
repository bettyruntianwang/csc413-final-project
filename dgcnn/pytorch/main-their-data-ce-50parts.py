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
from torch_model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from util import cross_entropy_loss, IOStream, get_part_point_cloud_from_label, cal_min_pairwise_seg_loss
import sklearn.metrics as metrics
import math
import h5py
from matplotlib import pyplot as plt
#import yaml
import sys
import time
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from visualization.visualize import plot3d_pts_in_camera_plane, plot3d_pts

VALIDATION_PERCENTAGE = 0.2

def _init_():
    if not os.path.exists('checkpoints_ce_loss_their_data_50parts'):
        os.makedirs('checkpoints_ce_loss_their_data_50parts')
    if not os.path.exists('checkpoints_ce_loss_their_data_50parts/'+args.exp_name):
        os.makedirs('checkpoints_ce_loss_their_data_50parts/'+args.exp_name)
    if not os.path.exists('checkpoints_ce_loss_their_data_50parts/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints_ce_loss_their_data_50parts/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints_ce_loss_their_data_50parts'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints_ce_loss_their_data_50parts' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints_ce_loss_their_data_50parts' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints_ce_loss_their_data_50parts' + '/' + args.exp_name + '/' + 'data.py.backup')

# Tianxu: return data and seg
def load_h5_data_their_data(h5_dir, file_num, num_points=2048):
    data = []
    label = []

    for fileid in range(file_num):
        f = h5py.File(os.path.join(h5_dir, F'ply_data_train{fileid}.h5'))
        
        data.append(f['data'][:]) # (2048, 2048, 3)
        label.append(f['pid'][:]) # (2048, 2048)
        
    data = torch.tensor(data, dtype=torch.float).reshape(5*2048, 2048, 3)   #number hard-coded!
    label = torch.tensor(label, dtype=torch.long).reshape(5*2048, 2048)     # number hard-coded!

    # # make labels in each object starting from 0
    # min_parts = torch.min(label, axis=1, keepdim=True)[0]
    # label = label - min_parts

    idx = np.arange(label.shape[0])
    np.random.shuffle(idx)
    data = data[idx, :][:, :num_points, :]
    label = label[idx, :][:, :num_points]
    return data, label

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
    # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(end_training_index)), drop_last=True)
    # test_loader  = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(range(end_training_index+1, data_size)), drop_last=True)

    train_loader = DataLoader(dataset, batch_size=batch_size)
    test_loader  = DataLoader(dataset, batch_size=batch_size)

    return train_loader, test_loader

def train(args, io):
    data_dir = os.path.join(BASE_DIR, '..', 'part_seg', 'hdf5_data_pytorch')
    #data_dir = '/home/tianxu/Desktop/pair-group/Thesis-project/dgcnn/dgcnn/tensorflow/part_seg/hdf5_data'

    data, label = load_h5_data_their_data(data_dir, 5, args.num_points)
    dataset = TensorDataset(data, label)

    train_loader, test_loader = get_data_loaders(dataset, args.batch_size)

    '''train_loader = DataLoader(data, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(, num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)'''

    device = torch.device("cuda" if args.use_cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, input_dim=3, part_num=50, num_points=args.num_points, batch_size=args.batch_size).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)

    if os.path.exists(args.model_path):
        io.cprint("Loading existing model...")
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            io.cprint("Existing model loaded")
        except:
            io.cprint("Can't load existing model, start from new model...")

    model.float()
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    #criterion = cal_min_pairwise_seg_loss # cross_entropy_loss

    train_loss_list = []
    train_acc_list = []
    train_balanced_acc_list = []
    test_loss_list = []
    test_acc_list = []
    test_balanced_acc_list = []
    max_test_acc = 0
    max_acc_epoch = 0
    min_test_loss = math.inf
    min_loss_epoch = 0

    starting_epoch = 0
    training_backup_filepath = F'checkpoints_ce_loss_their_data_50parts/{args.exp_name}/models/training_backup.txt'
    if os.path.exists(training_backup_filepath):
        try:
            with open(training_backup_filepath, 'r') as f:
                starting_epoch = int(f.readline()) + 1
                if starting_epoch >= args.epochs - 1:
                    starting_epoch = 0
                else:
                    max_test_acc = float(f.readline())
                    min_test_loss = float(f.readline())
        except:
            io.cprint("Error when reading epoch record file")

    io.cprint(F"Starting from epoch {starting_epoch}")
    for epoch in range(starting_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        start_time = time.time()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            # data: batch_size x point_num x 3
            # label: batch_size x point_num

            batch_size = data.shape[0]
            opt.zero_grad()
            logits = model(data.permute(0, 2, 1))

            # TODO: update for cross entropy
            #loss, permuted_labels = criterion(logits, label)
            min_loss = cross_entropy_loss(logits, label)
            min_loss.backward()

            opt.step()
            preds = logits.max(dim=2)[1]
            count += batch_size
            train_loss += min_loss.item() * batch_size
            train_true.append(label.cpu().view(-1).numpy())
            train_pred.append(preds.detach().view(-1).cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        train_loss = train_loss*1.0/count
        train_loss_list.append(train_loss)

        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_acc_list.append(train_acc)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch,
                                                            train_loss,
                                                            train_acc)
        io.cprint(outstr)

        scheduler.step()

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            batch_size = data.shape[0]
            logits = model(data.permute(0, 2, 1))
            
            loss = cross_entropy_loss(logits, label)
            preds = logits.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().view(-1).numpy())
            test_pred.append(preds.detach().cpu().view(-1).numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)

        test_loss = test_loss*1.0/count
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_acc_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints_ce_loss_their_data_50parts/%s/models/model.h5' % args.exp_name)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            min_loss_epoch = epoch
        
        end_time = time.time()
        time_per_epoch = end_time - start_time
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, total time: %.6f s\n' % (epoch,
                                                                                test_loss,
                                                                                test_acc,
                                                                                time_per_epoch)
        io.cprint(outstr)

        with open(training_backup_filepath, 'w') as f:
            f.write(str(epoch) + '\n')
            f.write(str(max_test_acc) + '\n')
            f.write(str(min_test_loss))       

    fig = plt.figure(figsize=(17, 10))

    loss_ax = fig.add_subplot(1, 2, 1)
    acc_ax = fig.add_subplot(1, 2, 2)

    loss_ax.plot(train_loss_list)
    loss_ax.plot(test_loss_list)
    loss_ax.set_title(F'Cross-entropy loss: \nMinimum test loss: {min_test_loss:.5f}(Epoch: {min_loss_epoch})')
    loss_ax.set_ylabel('loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.legend([F'train', \
                F'test'], loc='upper right')

    acc_ax.plot(train_acc_list)
    acc_ax.plot(test_acc_list)
    acc_ax.set_title(F'Accuracy: \nMaximum test accuracy: {max_test_acc:.5f}(Epoch: {max_acc_epoch})')
    acc_ax.set_ylabel('acc')
    acc_ax.set_xlabel('epoch')
    acc_ax.legend([F'train', \
                F'test'], loc='upper right')
    #plt.show()
    fig.savefig('./log_ce_loss_their_data-50parts/model_loss_acc.png')

def test(args, io):
    data_dir = os.path.join(BASE_DIR, '..', 'dataset', 'hdf5-Sapien', 'cabinets')
    total_objects = os.listdir(data_dir)
    np.random.shuffle(total_objects)

    #object_ids, step_ids, data, label, step_counts, part_counts = load_(data_dir, os.listdir(data_dir), args.num_points)
    data, label = load_h5_data_their_data(data_dir, os.listdir(data_dir), args.num_points)

    # for i, object_id in enumerate(object_ids):
    #     object_id = object_id.item()
    #     yml_load_path = os.path.join(BASE_DIR, '..', 'dataset', 'render-Sapien', 'cabinets', str(object_id), 'pose-transformation.yml')                                                     
    #     with open(yml_load_path, 'r') as f:
    #         yaml_dict = yaml.load(f)
    #     proj_matrix = np.array(yaml_dict['projMat']).reshape(4, 4)
    #     #view_matrix = np.array(yaml_dict['view_matrix_cam_1']).reshape(4, 4)
    #     #world_to_image_matrix = np.dot(view_matrix, proj_matrix)
    #     #pass_once(args, data[i], object_id=object_id, step_id=step_ids[i], project_to_plane=True, transform_matrix=proj_matrix)
    #     pass_once(args, data[i], object_id=object_id, step_count=step_counts[i], project_to_plane=True, transform_matrix=proj_matrix)

    dataset = TensorDataset(data, label)

    _, test_loader = get_data_loaders(dataset, batch_size=args.batch_size, val_percentage=1)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()
    test_acc = 0.0
    #count = 0.0
    test_pred = []
    test_true = []
    for data, label, step_counts, part_counts in test_loader:
        data, label = data.to(device), label.to(device)

        # remove padded values in label and data, then vstack them
        max_step_size = label.shape[1]
        data = data.view(data.shape[0]*data.shape[1], -1, 3)
        label = label.view(label.shape[0]*label.shape[1], -1)
        preserved_indices = torch.unique((label != -1).nonzero()[:, 0])
        data = data[preserved_indices, :, :]
        label = label[preserved_indices, :]
        part_counts = part_counts.unsqueeze(1).repeat(1, max_step_size).view(-1)[preserved_indices]

        permuted_data = data.permute(0, 2, 1)
        batch_size = permuted_data.size()[0]
        logits = model(permuted_data)
        
        loss, permuted_labels = cal_min_pairwise_seg_loss(logits, label, part_counts, step_counts)
        preds = logits.max(dim=2)[1]
        #count += batch_size
        #test_loss += loss.item() * batch_size
        test_true.append(permuted_labels.cpu().view(-1).numpy())
        test_pred.append(preds.detach().cpu().view(-1).numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f'%(test_acc)
    io.cprint(outstr)

def pass_once(args, points, object_id=0, step_count=13, max_part_count=10, project_to_plane=False, transform_matrix=None):
    output_dir = os.path.join(BASE_DIR, '..', 'visualize_output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_object_dir = os.path.join(output_dir, str(object_id))
    if not os.path.exists(output_object_dir):
        os.mkdir(output_object_dir)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()

    permuted_data = points.permute(0, 2, 1)
    logits = model(permuted_data)

    for step_id in range(step_count):
        step_preds = logits[step_id, :, :].max(dim=1)[1]
        step_points = points[step_id, :, :]
        part_points = get_part_point_cloud_from_label(step_points, step_preds, max_part_count)

        if project_to_plane:
            plot3d_pts_in_camera_plane(part_points, transform_matrix,
                                        pts_name=[f'part {i}: {part_points[i].shape[0]} pts' for i in range(max_part_count)], 
                                        title_name=F'object {object_id} step {step_id}',
                                        show_fig=False, save_fig=True,
                                        save_path=output_object_dir, filename=F'step {step_id}', s=10)
        else:
            plot3d_pts([part_points], pts_name=[[f'part {i}: {part_points[i].shape[0]} pts' for i in range(max_part_count)]], 
                                        title_name=F'object {object_id} step {step_id}',
                                        sub_name=F'step {step_id}', show_fig=True, save_fig=True,
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
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
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

    io = IOStream('checkpoints_ce_loss_their_data_50parts/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.model_path = os.path.join('checkpoints_ce_loss_their_data_50parts', args.exp_name, 'models', 'model.h5')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    args.eval = False
    if not args.eval:
        if not os.path.exists('./log_ce_loss_their_data-50parts'):
            os.mkdir('./log_ce_loss_their_data-50parts')
        train(args, io)
    else:
        test(args, io)
