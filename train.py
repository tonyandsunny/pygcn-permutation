from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas
import torch
import glob
import os
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from torch.autograd import Variable
from utils import load_citation,accuracy,load_data, load_corpus
from models import GCN
import visdom

vis = visdom.Visdom(env = 'loss_lines', port=8097)

#Training settings

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help="Disables CUDA training")
parser.add_argument('--fastmode', action='store_true', default=False,
                    help = "Validate during training pass")
parser.add_argument('--seed', type=int, default=42,help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learining rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay(L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1-keep probability)')
parser.add_argument('--dataset', type=str, default='nell.0.001', help='Dataset to use')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                                'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--rate1', type=float, default=0, help='features pertubation percentage')
parser.add_argument('--rate2', type=float, default=0, help='adj pertubation percentage')
parser.add_argument('--lambda1', type=float, default=0, help='The weight of features pertubation ')
parser.add_argument('--lambda2', type=float, default=0, help='The weight of adj pertubation')
parser.add_argument('--tuned', action='store_true', help='use to tune hyperparameter')
parser.add_argument('--from_random', type=int, default=0, help='Number of drop to train')
parser.add_argument('--model',type=str, default='gcn', help='[gcn |SGC]')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value of leaky_relu' )
parser.add_argument('--patience', type=int, default=100, help='patience ')
parser.add_argument('--best_epoch', type=int, default=0, help='loss val early_stop best epochs')
parser.add_argument('--run_time', type=int, default=1, help = 'run code times')
parser.add_argument('--early_stop', type=int, default=5, help='Tolerance for early stopping (# of epochs).')

args = parser.parse_args()
def gen_seeds(size: int = None) -> np.ndarray:
    max_uint32 = np.iinfo(np.uint32).max#编程中有时候需要一个初始极大值（或极小值）作为temp，
    # 当然可以自定义设置为10000（whatever），不过python中有一个值可以代替之。即，我们定义一个初始的极大值
    return np.random.randint(
            max_uint32 + 1, size=size, dtype=np.uint32)#产生的随机数在[0,max_uint32]之间
def train():
    acc_test_list = []
    for ii in range(args.run_time):
        #seed = gen_seeds()
        seed = args.seed
        print("dataset:{}, epochs:{}, weight_decay:{},lr:{},dropout:{},seed:{}, alpha:{}, features_perturbation: rate1:{},lambda1:{}; adj_pertubation: rate2:{},lambda2:{}".format(
            args.dataset, args.epochs, args.weight_decay, args.lr, args.dropout, seed, args.alpha, args.rate1, args.lambda1, args.rate2,args.lambda2))
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed(seed)

        if args.dataset == "R8":
             adj, features, labels, idx_train, idx_val, idx_test = load_corpus(args.dataset, args.normalization,args.cuda)
        else:
            adj, features, labels, idx_train, idx_val, idx_test, indices = load_citation(args.dataset, args.normalization, args.cuda)

        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




        global x,y,z
        x,y,z=0,0,0
        win = vis.line(X=np.array([x]),
                       Y=np.array([y]),
                       opts=dict(title='loss_CE'))
        global b_1,b_2
        b_0,b_1,b_2=0,0,0
        win_b0 = vis.line(X=np.array([x]),
                          Y=np.array([b_0]),
                          opts=dict(title='b'))
        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            output = F.log_softmax(output, dim=1)
            loss_CE = F.nll_loss(output[idx_train], labels[idx_train])
            #l2_reg = sum(torch.sum(param ** 2) for param in model.reg_params)
            acc_train = accuracy(output[idx_train], labels[idx_train])
            x = epoch
            y = loss_CE.detach().cpu().numpy()
            vis.line(X=np.array([x]),
                     Y=np.array([y]),
                     win=win,
                     update='append')

            #loss_train = loss_CE + args.weight_decay /2 *l2_reg
            loss_train = loss_CE
            loss_train.backward()
            optimizer.step()

            if not args.fastmode:
                model.eval()
                output = model(features, adj)
                output = F.log_softmax(output, dim=1)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if ii == 0:
                if epoch%10==0:

                    print('Epoch: {:04d}'.format(epoch + 1),
                              'loss_train: {:.4f}'.format(loss_train.item()),
                              'loss_CE: {:.4f}'.format(loss_CE.item()),
                              'acc_train: {:.4f}'.format(acc_train.item()),
                              #'loss_0: {:.4f}'.format(loss_0.item()),

                              # 'loss_fx: {:.4f}'.format(loss_fx.item()),
                              # 'loss_logfx: {:.4f}'.format(loss_logfx.item()),
                              # 'acc_train: {:.4f}'.format(acc_train.item()),
                              'loss_val: {:.4f}'.format(loss_val.item()),
                              'acc_val: {:.4f}'.format(acc_val.item()),
                              'time: {:.4f}s'.format(time.time() - t))
            return loss_val.item()
        def test():
            model.eval()

            output = model(features,adj)

            output = F.log_softmax(output, dim=1)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print("Test set results:",
              "loss_test= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
            return acc_test

        t_total = time.time()
        loss_values = []
        # bad_counter = 0
        # best = args.epochs+1
        # best_epoch=0
        # for epoch in range(args.epochs):
        #
        #     loss_values.append(train(epoch))
        #     torch.save(model.state_dict(), './checkpoints/{}/{}.pkl'.format(args.dataset, epoch))
        #     if loss_values[-1] < best:
        #         best = loss_values[-1]
        #         best_epoch = epoch
        #         bad_counter = 0
        #     else:
        #         bad_counter += 1
        #     if bad_counter == args.patience:
        #         break
        #     files = glob.glob('./checkpoints/{}/*.pkl'.format(args.dataset))  # =[/checkpoints/{}/{}/0.pkl]
        #     for file in files:
        #         epoch_nb = int(file.split('.')[-2].split('/')[-1])
        #         if epoch_nb < best_epoch:
        #             os.remove(file)
        # files = glob.glob('./checkpoints/{}/*.pkl'.format(args.dataset))
        # for file in files:
        #     epoch_nb = int(file.split('.')[-2].split('/')[-1])
        #     if epoch_nb > best_epoch:
        #         os.remove(file)
        # print("Optimization Finished!")
        # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        #
        # # Testing
        # print('Loading {}th epoch'.format(best_epoch))
        # model.load_state_dict(torch.load('./checkpoints/{}/{}.pkl'.format(args.dataset, best_epoch)))
        for epoch in range(args.epochs):
            loss_values.append(train(epoch))
            if epoch>args.early_stop and loss_values[-1] > np.mean(loss_values[-(args.early_stop+1):-1]):
                print("Early stopping...")
                break
            #train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        acc_test = test()
        acc_test_list.append(acc_test)
        acc_test = acc_test.view(1, 1)
        acc_test = acc_test.cpu().numpy()

        with open("./results/{}/{}.txt".format(args.dataset, args.dataset), 'a') as f:
            # f.write("不丢弃dropout,加上b= {:.07f}×(output-output_1),epoch : {:04d},weight_decacy={},系数lam ={:.07f}".format
            f.write("dataset{} epoch : {:04d},weight_decacy={}, lr{},seed{},dropout{},features_perturbation: rate1:{},lambda1:{}; adj_pertubation: rate2:{},lambda2:{}".format
                    (args.dataset, args.epochs, args.weight_decay, args.lr, seed, args.dropout,args.rate1,args.lambda1,args.rate2,args.lambda2))
            np.savetxt(f, acc_test, fmt="%.6f")
    acc_test_list = torch.FloatTensor(acc_test_list)
    acc_test_std = torch.std(acc_test_list)
    avg_test = torch.mean(acc_test_list)

    avg_test = avg_test.view(1, 1)
    avg_test = avg_test.cpu().numpy()
    acc_test_std = acc_test_std.view(1, 1)
    acc_test_std = acc_test_std.cpu().numpy()
    print("总共做类{}次实验，平均值为：{:.04f}".format(args.run_time, avg_test.item()))
    print("总共做类{}次实验，误差值为：{:.04f}".format(args.run_time,acc_test_std.item()))
    with open("./results/{}/{}.txt".format(args.dataset, args.dataset), 'a') as f:
        f.write("总共做类{}次实验，平均值为：{:.04f}\n".format(args.run_time, avg_test.item()))
        f.write("总共做类{}次实验，误差值为：{:.04f}\n".format(args.run_time, acc_test_std.item()))
        # np.savetxt(f, avg_test, fmt="%.18f")
        # args.epochs = args.epochs+100

    # args.epochs = 100


#train()
while args.lr <= 0.1 and args.lr >= 0.01:
     train()
     args.lr += 0.01
