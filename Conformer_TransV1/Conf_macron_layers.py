#! /usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from torch.autograd import Variable


#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer. FFN(x) = max(0, xW1 + b1)W2 + b2 """

    def __init__(self, d_model, d_ff, dropout=0.1,activation=torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_ff)
        self.activation = activation
    def forward(self, x):
        output = self.activation(self.layer_norm(self.w_1(x)))
        output = self.dropout(output)
        output = self.w_2(output)
        return output

##############################################################################################################


