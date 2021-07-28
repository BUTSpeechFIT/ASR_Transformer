#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import torch


#*************************************************************************************************************************
####### Loading the Parser and default arguments
#import pdb;pdb.set_trace()

#sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Trans_V1')
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
import Transformer_arg
from Transformer_arg import parser
args = parser.parse_args()

#************************
import Set_gpus
from Set_gpus import Set_gpu
if args.gpu:
    Set_gpu()
#***********************

import numpy as np
import fileinput
import json
import random
from itertools import chain
from numpy.random import permutation
##------------------------------------------------------------------
#import torch
from torch.autograd import Variable
#----------------------------------------
import torch.nn as nn
from torch import autograd, nn, optim
os.environ['PYTHONUNBUFFERED'] = '0'
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from random import shuffle
from statistics import mean
import glob

###save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)
print(args)
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
# #####setting the gpus in the gpu cluster
# #**********************************
#import Set_gpus
#from Set_gpus import Set_gpu
#if args.gpu:
#    Set_gpu()
    
###----------------------------------------
from Dataloader_for_AM_v2 import DataLoader
from utils__ import weights_init,reduce_learning_rate,read_as_list,gaussian_noise,plotting
#==============================================================
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
#from TRANSFORMER_ASR_V1 import Transformer
#from Initializing_Transformer_ASR import Initialize_Att_model
#from Transformer_Training_loop import train_val_model
from Load_sp_model import Load_sp_models
#=============================================================
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)


#=============================================================
def main():
        ##Load setpiece models for Dataloaders
        Word_model=Load_sp_models(args.Word_model_path)
        Char_model=Load_sp_models(args.Char_model_path)
        ###initilize the model
        #model,optimizer=Initialize_Att_model(args)
        #============================================================
        #------------------------------------------------------------  
        #
        train_gen = DataLoader(files=glob.glob(args.data_dir + "train_scp_splits/*"), 
                                max_batch_label_len=args.max_batch_label_len,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=args.max_feat_len,
                                max_label_len=args.max_label_len,
                                Word_model=Word_model,
                                Char_model=Char_model,
                                apply_cmvn=int(args.apply_cmvn))


        dev_gen = DataLoader(files=glob.glob(args.data_dir + "dev_splits/*"),
                                max_batch_label_len=args.max_batch_label_len,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=5000,
                                max_label_len=1000,
                                Word_model=Word_model,
                                Char_model=Char_model,
                                apply_cmvn=int(args.apply_cmvn))


        #======================================
        for epoch in range(args.nepochs):
            ##start of the epoch
            tr_CER=[]; tr_BPE_CER=[]; L_train_cost=[]
            
            validate_interval = int(args.validate_interval * args.accm_grad) if args.accm_grad>0 else args.validate_interval
            for trs_no in range(validate_interval):
                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"
                B2 = dev_gen.next()
                assert B2 is not None, "None should never come out of the DataLoader"

                print(trs_no, B1.get('smp_feat').shape,B1.get('smp_word_label').shape,B2.get('smp_feat').shape,B2.get('smp_word_label').shape)
                

#=======================================================
#=============================================================================================
if __name__ == '__main__':
    main()



