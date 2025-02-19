#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--addtime', type=int, default=5, help="when add attack")
    parser.add_argument('--frac', type=float, default=0.25, help="the fraction of clients: C")
    parser.add_argument('--malicious', type=int, default=11, help="the number of malicious")
    parser.add_argument('--epsilon', type=float, default=0.1, help="the number of malicious")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    parser.add_argument('--model', type=str, default='cnn', help='model name')   #### cnn
    parser.add_argument('--attack', type=str, default='xie', help='attack name')
    parser.add_argument('--Agg',  default='Fedavg', help='Krum, Fedavg')
    parser.add_argument('--usegrad', action='store_true', help='use grad')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")                    ####mnist
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')                         ####
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")                          ####CPU
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    #LDP parameter
    parser.add_argument('--C', type=int, default=3, help="Gradient norm bound")
    parser.add_argument('--sigma', type=int, default=7, help="Noise scale, standard variance in gaussian")

    args = parser.parse_args()
    return args
