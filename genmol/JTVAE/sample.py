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

from jvae_model import *

path = "savedmodel.pth"
model=JTNNVAE(vocab, int(450), int(56), int(20), int(3))
model.load_state_dict(torch.load(path))
torch.manual_seed(0)
print("Molecules generated")
for i in range(10):
    print(model.sample_prior())


