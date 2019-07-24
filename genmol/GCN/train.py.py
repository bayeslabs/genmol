#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hyperparameter
seed=42,
epochs=200,
lr=0.01,
weight_decay=16,
hidden=16,
dropout=0.5,


# In[ ]:


from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#if args.cuda:
    #torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, train, val, test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

#if args.cuda:
    #model.cuda()
    #features = features.cuda()
    #adj = adj.cuda()
    #labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()


def train(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[train], labels[train])
    acc_train = accuracy(output[train], labels[train])
    loss_train.backward()
    optimizer.step()

    #if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[val], labels[val])
    acc_val = accuracy(output[val], labels[val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))



