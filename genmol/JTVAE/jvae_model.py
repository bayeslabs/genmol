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

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


def get_molecule(node):
    return Chem.MolFromSmiles(node.smiles)


class Vocab(object):
    benzynes = ['C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CC=NN=C1', 'C1=CN=CC=N1', 'C1=CN=CN=C1', 'C1=CN=NC=N1', 'C1=CN=NN=C1', 'C1=NC=NC=N1', 'C1=NN=CN=N1']
    penzynes = ['C1=C[NH]C=C1', 'C1=C[NH]C=N1', 'C1=C[NH]N=C1', 'C1=C[NH]N=N1', 'C1=COC=C1', 'C1=COC=N1', 'C1=CON=C1', 'C1=CSC=C1', 'C1=CSC=N1', 'C1=CSN=C1', 'C1=CSN=N1', 'C1=NN=C[NH]1', 'C1=NN=CO1', 'C1=NN=CS1', 'C1=N[NH]C=N1', 'C1=N[NH]N=C1', 'C1=N[NH]N=N1', 'C1=NN=N[NH]1', 'C1=NN=NS1', 'C1=NOC=N1', 'C1=NON=C1', 'C1=NSC=N1', 'C1=NSN=C1']

    def __init__(self, smiles_list,all_trees):
        list_d=[]

        for j in range(0,len(all_trees)):
            x=[]
            x=all_trees[j].nodes

            for i in range(0,len(x)):
                m=get_molecule(x[i])
                m1=Chem.MolToSmiles(m,kekuleSmiles=False)
                list_d.append(m1)

        list_f=list(dict.fromkeys(list_d))
        smiles_f=smiles_list+list_f
        
        self.vocab = smiles_f
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        Vocab.benzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6] + ['C1=CCNCC1']
        Vocab.penzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5] + ['C1=NCCN1','C1=NNCC1']


    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)



def create_variable(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x,sum_h], dim=1)
    z = torch.sigmoid(W_z(z_input))

    r_1 = W_r(x).view(-1,1,hidden_size)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x,sum_gated_h], dim=1)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h



ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM1 = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM1 = 5 + 6
MAX_NB1 = 6

def onek_encoding_unk1(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features1(atom):
    return torch.Tensor(onek_encoding_unk1(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk1(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk1(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk1(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features1(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk1(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

class MPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = int(hidden_size)
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM1 + BOND_FDIM1, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM1 + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope):
        fatoms = create_variable(fatoms)
        fbonds = create_variable(fbonds)
        agraph = create_variable(agraph)
        bgraph = create_variable(bgraph)

        binput = self.W_i(fbonds)
        message = F.relu(binput)

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        max_len = max([x for _,x in scope])
        batch_vecs = []
        for st,le in scope:
            cur_vecs = atom_hiddens[st : st + le].mean(dim=0)
            batch_vecs.append( cur_vecs )

        mol_vecs = torch.stack(batch_vecs, dim=0)
        return mol_vecs 

    @staticmethod
    def tensorize(mol_batch):
        padding = torch.zeros(ATOM_FDIM1 + BOND_FDIM1)
        fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
        in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            n_atoms = mol.GetNumAtoms()
            
            for atom in mol.GetAtoms():
                fatoms.append( atom_features1(atom) )
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds) 
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bond_features1(bond)], 0) )
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bond_features1(bond)], 0) )
                in_bonds[x].append(b)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms,MAX_NB1).long()
        bgraph = torch.zeros(total_bonds,MAX_NB1).long()

        for a in range(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(1, total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1,i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)



ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 15

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])

class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = int(hidden_size)
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope, tree_message): #tree_message[0] == vec(0)
        fatoms = create_variable(fatoms)
        fbonds = create_variable(fbonds)
        agraph = create_variable(agraph)
        bgraph = create_variable(bgraph)

        binput = self.W_i(fbonds)
        graph_message = F.relu(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message,graph_message], dim=0) 
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1) #assuming tree_message[0] == vec(0)
            nei_message = self.W_h(nei_message)
            graph_message = F.relu(binput + nei_message)

        message = torch.cat([tree_message,graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

    @staticmethod
    def tensorize(cand_batch, mess_dict):
        fatoms,fbonds = [],[] 
        in_bonds,all_bonds = [],[] 
        total_atoms = 0
        total_mess = len(mess_dict) + 1 #must include vec(0) padding
        scope = []

        for smiles,all_nodes,ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol) #The original jtnn version kekulizes. Need to revisit why it is necessary
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx
            for atom in mol.GetAtoms():
                
                fatoms.append( atom_features(atom) )
                in_bonds.append([]) 
        
            for bond in mol.GetBonds():
                
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                #Here x_nid,y_nid could be 0
                x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = total_mess + len(all_bonds)  #bond idx offseted by total_mess
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
                in_bonds[y].append(b)

                b = total_mess + len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid,y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid,y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid,x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid,x_bid)]
                        in_bonds[x].append(mess_idx)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms
        
        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()

        for a in range(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]): #b2 is offseted by total_mess
                if b2 < total_mess or all_bonds[b2-total_mess][0] != y:
                    bgraph[b1,i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)



def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        dfs(stack, y, x.idx)
        stack.append( (y,x,0) )

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True

def can_assemble(node_x, node_y):
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands,aroma_scores = enum_assemble(node_x, neighbors)
    return len(cands) > 0# and sum(aroma_scores) >= 0



MAX_NB0 = 15
MAX_DECODE_LEN0 = 100

class JTNNDecoder(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, embedding):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding
        latent_size=int(latent_size)
        #GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        #Word Prediction Weights 
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        #Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        #Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)


        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        tree_contexts = x_tree_vecs.index_select(0, contexts)
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        output_vec = F.relu( V(input_vec) )
        return V_o(output_vec)

    def forward(self, mol_batch, x_tree_vecs):
        pred_hiddens,pred_contexts,pred_targets = [],[],[]
        stop_hiddens,stop_contexts,stop_targets = [],[],[]
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        #Predict Root
        batch_size = len(mol_batch)
        pred_hiddens.append(create_variable(torch.zeros(len(mol_batch),self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])

        pred_contexts.append( create_variable( torch.LongTensor(range(batch_size)) ) )

        max_iter = max([len(tr) for tr in traces])
        padding = create_variable(torch.zeros(self.hidden_size), False)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei,cur_o_nei = [],[]

            for node_x, real_y, _ in prop_list:
                #Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB0 - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB0 - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                #Current clique embedding
                cur_x.append(node_x.wid)


            #Clique embedding
            cur_x = create_variable(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x) 
            
            #Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB0,self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            #Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB0,self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(prop_list):
                node_x,node_y,direction = m
                x,y = node_x.idx,node_y.idx
                h[(x,y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i) 
                stop_target.append(direction)

            #Hidden states for stop prediction
            cur_batch = create_variable(torch.LongTensor(batch_list))
            stop_hidden = torch.cat([cur_x,cur_o], dim=1)
            stop_hiddens.append( stop_hidden )
            stop_contexts.append( cur_batch )
            stop_targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_variable(torch.LongTensor(batch_list))
                pred_contexts.append( cur_batch )

                cur_pred = create_variable(torch.LongTensor(pred_list))
                pred_hiddens.append( new_h.index_select(0, cur_pred) )
                pred_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB0 - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_variable(torch.LongTensor(cur_x))
        cur_x = self.embedding(cur_x) 
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB0,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x,cur_o], dim=1)
        stop_hiddens.append( stop_hidden )
        stop_contexts.append( create_variable( torch.LongTensor(range(batch_size)) ) )
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'word')
        pred_targets = create_variable(torch.LongTensor(pred_targets))
        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        #Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu( self.U_i(stop_hiddens) )
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, x_tree_vecs, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = create_variable(torch.Tensor(stop_targets))
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()
    
    def decode(self, x_tree_vecs, prob_decode):
        assert x_tree_vecs.size(0) == 1

        stack = []
        init_hiddens = create_variable( torch.zeros(1, self.hidden_size) )
        zero_pad = create_variable(torch.zeros(1,1,self.hidden_size))
        contexts = create_variable( torch.LongTensor(1).zero_() )

        #Root Prediction
        root_score = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'word')
        _,root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append( (root, self.vocab.get_slots(root.wid)) )

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN0):
            node_x,fa_slot = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_variable(torch.LongTensor([node_x.wid]))
            cur_x = self.embedding(cur_x)

            #Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
            stop_hiddens = F.relu( self.U_i(stop_hiddens) )
            stop_score = self.aggregate(stop_hiddens, contexts, x_tree_vecs, 'stop')
            
            if prob_decode:
                backtrack = (torch.bernoulli( torch.sigmoid(stop_score) ).item() == 0)
            else:
                backtrack = (stop_score.item() < 0) 

            if not backtrack: #Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.aggregate(new_h, contexts, x_tree_vecs, 'word')

                if prob_decode:
                    sort_wid = torch.multinomial(F.softmax(pred_score, dim=1).squeeze(), 5)
                else:
                    _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y,next_slots) )
                    all_nodes.append(node_y)

            if backtrack: #Backtrack, use if instead of else
                if len(stack) == 1: 
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes




class JTNNEncoder(nn.Module):

    def __init__(self, hidden_size, depth, embedding):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = int(hidden_size)
        self.depth = depth

        self.embedding = embedding
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope):
        fnode = create_variable(fnode)
        fmess = create_variable(fmess)
        node_graph = create_variable(node_graph)
        mess_graph = create_variable(mess_graph)
        messages = create_variable(torch.zeros(mess_graph.size(0), self.hidden_size))

        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        node_vecs = self.outputNN(node_vecs)

        max_len = max([x for _,x in scope])
        batch_vecs = []
        for st,le in scope:
            cur_vecs = node_vecs[st] #Root is the first node
            batch_vecs.append( cur_vecs )

        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):

        node_batch = [] 
        scope = []
        for tree in tree_batch:
            scope.append( (len(node_batch), len(tree.nodes)) )
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)
    
    @staticmethod
    def tensorize_nodes(node_batch, scope):
      
        messages,mess_dict = [None],{}
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)
            for y in x.neighbors:
                mess_dict[(x.idx,y.idx)] = len(messages)
                messages.append( (x,y) )

        node_graph = [[] for i in range(len(node_batch))]
        mess_graph = [[] for i in range(len(messages))]
        fmess = [0] * len(messages)

        for x,y in messages[1:]:
            mid1 = mess_dict[(x.idx,y.idx)]
            fmess[mid1] = x.idx 
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx: continue
                mid2 = mess_dict[(y.idx,z.idx)]
                mess_graph[mid2].append(mid1)

        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)
        node_graph = torch.LongTensor(node_graph)
        fmess = torch.LongTensor(fmess)
        fnode = torch.LongTensor(fnode)
        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict

class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0 #first vector is padding
        mask = create_variable(mask)
        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = torch.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = torch.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h







class JTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab

        self.hidden_size = int(hidden_size)
        self.latent_size = latent_size = latent_size / 2 #Tree and Mol has two vectors
        self.latent_size=int(self.latent_size)
        self.jtnn = JTNNEncoder(int(hidden_size),int(depthT), nn.Embedding(780,450))
        self.decoder = JTNNDecoder(vocab, int(hidden_size), int(latent_size), nn.Embedding(780,450))

        self.jtmpn = JTMPN(int(hidden_size), int(depthG))
        self.mpn = MPN(int(hidden_size), int(depthG))

        self.A_assm = nn.Linear(int(latent_size), int(hidden_size), bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(int(hidden_size), int(latent_size))
        self.T_var = nn.Linear(int(hidden_size), int(latent_size))
        self.G_mean = nn.Linear(int(hidden_size), int(latent_size))
        self.G_var = nn.Linear(int(hidden_size), int(latent_size))

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_variable(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size)
        z_mol = torch.randn(1, self.latent_size)
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch, beta):
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder= x_batch
        #ncoding the graph and tree
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
       
        z_mol_vecs,mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
       
        kl_div = tree_kl + mol_kl
        #Decoding the tree
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        #Decoding the graph and assembling the graph
        assm_loss, assm_acc = self.assm(x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess)

        return word_loss + topo_loss + assm_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc, assm_acc

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder,batch_idx = jtmpn_holder
        fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
        batch_idx = create_variable(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs) #bilinear
        scores = torch.bmm(
                x_mol_vecs.unsqueeze(1),
                cand_vecs.unsqueeze(-1)
        ).squeeze()
        
        cnt,tot,acc = 0,0,0
        all_loss = []
        for i,mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_variable(torch.LongTensor([label]))
                all_loss.append( self.assm_loss(cur_score.view(1,-1), label) )
        
        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root,pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.smiles

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze() #bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol,_ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol,pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: 
            return None



        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol

#Reading the input
vocab = [x.strip("\r\n ") for x in open('train.txt')] 
#Building the vocabulary
vocab = Vocab(vocab,mol_trees)

#Defining the model
model = JTNNVAE(vocab, int(450), int(56), int(20), int(3))

print("Model")
print(model)




