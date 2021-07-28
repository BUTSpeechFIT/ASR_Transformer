#! /usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable
import sys

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/Conformer_TransV1')
import Transformer_arg
from Transformer_arg import parser
args = parser.parse_args()

print(args)

from Conf_Encoder import Encoder
from Trans_Decoder import Decoder
encoder = Encoder(args,MT_flag=False)

#
#
Feat = torch.rand(5,53,256)
Labels = torch.randint(0,50,(5,53))
breakpoint()

A = encoder(Feat)
Dec = Decoder(args)
Dec(Labels,A)
