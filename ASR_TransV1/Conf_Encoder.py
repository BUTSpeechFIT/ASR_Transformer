#! /usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable

from Trans_conv_layers import Conv_2D_Layers

from Conf_RelMHA import RelPositionMultiHeadedAttention
from Conf_RelPos import RelPositionalEncoding
from Conf_conv_layers import ConvolutionModule
from Conf_macron_layers import PositionwiseFeedForward
from Conf_swish_Act import Swish



from Trans_utilities import get_attn_key_pad_mask, get_subsequent_mask, get_attn_pad_mask_encoder, get_attn_pad_mask,get_encoder_non_pad_mask, get_decoder_non_pad_mask
#from Trans_MHA import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, channels, kernel_size, dropout=0.1,ff_dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.ff_scale = 0.5
        self.feed_forward_macaron = True
        self.normalize_before = True
        self.activation = Swish() 
 
        self.slf_attn = RelPositionMultiHeadedAttention(n_head, d_model,dropout=dropout)
        if self.feed_forward_macaron:
                self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=ff_dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner, dropout=ff_dropout)
        self.conv_module  = ConvolutionModule(channels, kernel_size,activation=self.activation)

        #self.Rel_pos = RelPositionalEncoding(d_model, dropout, max_len=5000)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, enc_output, pos_emb, non_pad_mask=None, slf_attn_mask=None):

        x=enc_output
        #------------------------------------------------------------------
        # macaron style feedforward
        if self.feed_forward_macaron:
            #residual = x
            #if self.normalize_before:
            nx = self.norm_ff1(x)
            x = x + self.ff_scale * self.dropout(self.pos_ffn1(nx))
        #--------------------------------------------------------------------
        ##self_attention
        nx = self.norm_attn(x)
        enc_output = self.slf_attn(nx, nx, nx, pos_emb, mask=slf_attn_mask)
        x = x + self.dropout(enc_output) 
        #--------------------------------------------------------------------
        ##conv_module
        nx = self.norm_attn(x)
        x = x + self.dropout(self.conv_module(nx))
        #--------------------------------------------------------------------
        nx = self.norm_ff2(x)
        x = x + self.dropout(self.pos_ffn2(nx))
        #
        #--------------------------------------------------------------------
        return x
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward. """

    def __init__(self, args, MT_flag):
        super(Encoder, self).__init__()

        # parameters
        self.d_input = args.encoder_dmodel
        self.n_layers = args.encoder_layers 
        self.n_head = args.encoder_heads
        self.d_model = args.encoder_dmodel
        self.d_inner = args.encoder_dinner
        self.channels = args.encoder_dmodel
        self.kernel_size = args.Conf_kernel_size

        self.d_k = int(self.d_model/self.n_head)
        self.d_v = int(self.d_model/self.n_head)
        self.dropout = args.encoder_dropout
        self.encoder_ff_dropout = args.encoder_ff_dropout
        self.pe_maxlen = args.pe_max_len
        self.xscale = math.sqrt(self.d_model)

        self.MT_flag = MT_flag
        #=======================================================
        # use linear transformation with layer norm to replace input embedding
        ###switches between ASR and MT modules
        if self.MT_flag:
            self.Src_model = Load_sp_models(args.Src_model_path)
            self.Src_model_tgts = int(self.Src_model.__len__())
            self.Src_model_vocab = self.Src_model_tgts + 4
            self.linear_in = nn.Embedding(self.Src_model_vocab, self.d_model)
        else:
            self.linear_in = nn.Linear(self.d_input, self.d_model)
        #========================================================

        self.layer_norm_in = nn.LayerNorm(self.d_model)
        self.positional_encoding = RelPositionalEncoding(self.d_model,self.dropout, max_len=5000)
        self.layer_stack = nn.ModuleList([EncoderLayer(self.d_model, self.d_inner, self.n_head, self.channels, self.kernel_size, dropout=self.dropout, ff_dropout=self.encoder_ff_dropout) for _ in range(self.n_layers)])
    def forward(self, padded_input, return_attns=False):
        """ Args: padded_input: N x Ti x D  input_lengths: N Returns: enc_output: N x Ti x H """ 

        enc_slf_attn_list = []      
        #Prepare masks  

        non_pad_mask=None;
        dec_enc_attn_mask=None;
        slf_attn_mask=None;
        slf_attn_mask_keypad=None

        padded_input_norm=self.layer_norm_in(self.linear_in(padded_input))
        enc_output_pl_embeding,pos_emb = self.positional_encoding(padded_input_norm)
        
        
        enc_output = padded_input_norm
        for enc_layer in self.layer_stack:

            enc_output = enc_layer(enc_output,pos_emb,non_pad_mask=non_pad_mask,slf_attn_mask=slf_attn_mask)
            enc_slf_attn=None

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        #
        return enc_output, enc_slf_attn
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
