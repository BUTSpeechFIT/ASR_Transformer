#! /usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable



MIN_VALUE = float(np.finfo(np.float32).min)
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.matmul(q, k.transpose(-2,-1))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask==0, MIN_VALUE)

        attn = self.softmax(attn)

        ##no self attn_dropout
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #================================
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        #================================

        q = self.w_qs(q).view(sz_b, -1, n_head, d_k)
        k = self.w_ks(k).view(sz_b, -1, n_head, d_k)
        v = self.w_vs(v).view(sz_b, -1, n_head, d_v)
        #================================
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,n_head,1, 1)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1,2).contiguous().view(sz_b,-1, n_head*d_v)
        output = self.fc(output)
        return output, attn


#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
     """Implement the positional encoding (PE) function. 
#     PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
#     PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
#     """

     def __init__(self, d_model, dropout,max_len=5000):
         super(PositionalEncoding, self).__init__()
         # Compute the positional encodings once in log space.
         self.dropout = dropout
         pe = torch.zeros(max_len, d_model, requires_grad=False)
         self.dropout=nn.Dropout(self.dropout)

         position = torch.arange(0, max_len).unsqueeze(1).float()
         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)

         self.x_scale = math.sqrt(d_model)
         pe = pe.unsqueeze(0)
         self.register_buffer('pe', pe)

     def forward(self, input):
         """ Args:   input: N x T x D  """
         length = input.size(1)
         PE=Variable(self.pe[:, :length],requires_grad=False)
         return self.dropout(input + PE)
#=============================================================================================================


#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer. FFN(x) = max(0, xW1 + b1)W2 + b2 """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_ff)

    def forward(self, x):
        output = F.relu(self.layer_norm(self.w_1(x)))
        output = self.dropout(output)
        output = self.w_2(output)
        return output

##############################################################################################################
