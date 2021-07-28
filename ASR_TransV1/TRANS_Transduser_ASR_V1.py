import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wtnrm
import numpy as np

from keras.preprocessing.sequence import pad_sequences

import sys
import kaldi_io
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
from CMVN import CMVN
from utils__ import weights_init,count_parameters

#--------------------------------------------------------------------------
class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention. """
    def __init__(self,args):
        super(Transformer, self).__init__()   
        self.label_smoothing = args.label_smoothing

        if args.conv_model=='Conv_2D_4Layers':
            print("using conv layers",)
            from Trans_conv_layers import Conv_2D_4Layers as Conv_2D_Layers
        else:
            from Trans_conv_layers import Conv_2D_Layers
        #print("using conv layers",Conv_2D_Layers)
        
        if args.encoder=='conformer':
           from Conf_Encoder import Encoder 
        else:
            from Trans_Encoder import Encoder

        #----------------------------------
        if args.decoder=='decoder':
            print("Nothing special")
        else:
            from Trans_Decoder import Decoder
        #----------------------------------

        self.conv_layers = Conv_2D_Layers(args=args)
        self.encoder = Encoder(args=args,MT_flag=False)
        self.decoder = Decoder(args=args)
    #----------------------------------
    def forward(self, padded_input,padded_target):
        ###conv layers
        conv_padded_input=self.conv_layers(padded_input)
        
        #General Transformer ASR model
        encoder_padded_outputs, *_ = self.encoder(conv_padded_input)
        #
        output_dict = self.decoder(padded_target, encoder_padded_outputs)
        return output_dict
    #=============================================================================================================
    #=============================================================================================================
    def predict(self, feat_path,args):
        print("went to the decoder loop")
        with torch.no_grad():

                #### read feature matrices 
                smp_feat=kaldi_io.read_mat(feat_path)
                smp_feat=CMVN(smp_feat)
                input=torch.from_numpy(smp_feat)       
                input = Variable(input.float(), requires_grad=False).double().float()
                input=input.unsqueeze(0)
                
                #print("args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen",args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen)
                ###conv layers
                conv_padded_input=self.conv_layers(input)

                #General Transformer ASR model
                encoder_padded_outputs, *_ = self.encoder(conv_padded_input)
                nbest_hyps,scoring_list = self.decoder.recognize_batch_beam_autoreg_LM_multi_hyp(encoder_padded_outputs,args.beam,args.Am_weight,args.gamma,args.LM_model,args.len_pen,args)
                #===================================================================================
                beam_len = nbest_hyps.size(0)
                hyp = {'score': 0.0, 'yseq': None,'state': None, 'alpha_i_list':None, 'Text_seq':None}
                #===============================================
                Output_dict=[]
                for I in range(beam_len):    
                    new_hyp={}
                    new_hyp['yseq'] = nbest_hyps[I]
                    new_hyp['score'] = scoring_list[I].sum()
                    new_hyp['Text_seq'] = self.decoder.get_charecters_for_sequences(nbest_hyps[I].unsqueeze(0))

                    new_hyp['state'] = hyp['state']
                    new_hyp['alpha_i_list'] = hyp['alpha_i_list']

                    Output_dict.append(new_hyp)
        return Output_dict
        #----------------------------------------------------------------
#=============================================================================================================
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, step_num, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        
        #present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = step_num
        self.reduction_factor=1
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))
        
        #print(lr,self.step_num ** (-0.5),self.step_num * self.warmup_steps ** (-1.5),self.reduction_factor)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_step_num(self, step_num):
        self.step_num=step_num

    def reduce_learning_rate(self, k):
        self.reduction_factor = self.reduction_factor*k
        #print(self.reduction_factor)
    
    def print_lr(self):
        present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return present_lr[0]


#==============================================================================================================
#---------------------------------------------------------------------------------------------------------------
