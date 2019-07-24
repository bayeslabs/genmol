import torch
import torch.nn as nn
import math, random, sys
from optparse import OptionParser
import pickle
import rdkit
import json
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
import copy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from collections import deque
import os, random
import torch.nn.functional as F
import pdb


from jvae_preprocess import *
from jvae_model import *


def set_batch_nodeID(mol_batch, vocab):
    tot = 0

    for mol_tree in mol_batch:

        for node in mol_tree.nodes:
            node.idx = tot
  
            s_to_m=Chem.MolFromSmiles(node.smiles)
            m_to_s=Chem.MolToSmiles(s_to_m,kekuleSmiles=False)
            node.wid = vocab.get_index(m_to_s)
        
            tot += 1


def tensorize_x(tree_batch, vocab,assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize_x(self.data[idx], self.vocab,assm=self.assm)



def get_loader(data_1,vocab):
    
    for i in range(0,len(data_1)):

        if True: 
            random.shuffle(data_1[i]) 

        batches=[]
        for j in range(0,len(data_1[i])):
            batches.append([])

        for j in range(0,len(data_1[i])):
            
            batches[j].append(data_1[i][j])

        dataset = MolTreeDataset(batches, vocab,True)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

        for b in dataloader:
            yield b

        del batches, dataset, dataloader
        



for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)


print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))



optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = 0.0
meters = np.zeros(4)
path = "savedmodel.pth"
print("Training")
#Training starts here...
for epoch in range(10):
	
    #Loading the data
    loader=get_loader(trees_data,vocab)
    
    for batch in loader:
        total_step += 1
        try:
            model.zero_grad()
            #Send the batch to the model
            loss, kl_div, wacc, tacc, sacc = model(batch, beta)
            #Backward propagation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),50.0)
            optimizer.step()
        except Exception as e:
            print(e)
            continue
   
        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

       
        torch.save(model.state_dict(), path)

        
        scheduler.step()
        #print("learning rate: %.6f" % scheduler.get_lr()[0])

        beta = min(1.0, beta + 0.002)

    print("Epoch :" + str(epoch))

