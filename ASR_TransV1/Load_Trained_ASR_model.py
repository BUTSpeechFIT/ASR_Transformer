#!/usr/bin/python
import sys
import os
import subprocess

from os.path import join, isdir, isfile
import torch
import json

import numpy as np
from torch import autograd, nn, optim
import torch.nn.functional as F

from argparse import Namespace









#**********
#Loading the Parser and default arguments
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')

import Transformer_arg
from Transformer_arg import parser


# from Load_sp_model import Load_sp_models
# from CMVN import CMVN
# from utils__ import plotting,read_as_list
# from user_defined_losses import compute_cer
# from Decoding_loop import get_cer_for_beam

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
from TRANSFORMER_ASR_V1 import Transformer
from Initializing_Transformer_ASR import Initialize_Att_model
from Stocasting_Weight_Addition import Stocasting_Weight_Addition
#-----------------------------------



def Load_Transformer_ASR_model(model_dir, SWA_random_tag, est_cpts=None):
        print(model_dir, SWA_random_tag)
        #=================================================================

        from Load_sp_model import Load_sp_models
        from Initializing_Transformer_ASR import Initialize_Att_model as Initialize_Trans_model
        from get_best_weights import get_best_weights
        from Stocasting_Weight_Addition import Stocasting_Weight_Addition
        #===================================================================

        SWA_random_tag = SWA_random_tag
        model_dir = model_dir

        model_path_name=join(model_dir,'model_architecture_')
        
        ###load the architecture if you have to load
        with open(model_path_name, 'r') as f:
                TEMP_args = json.load(f)

        args = Namespace(**TEMP_args)
        args.gpu=0
        
        ##to swith from using deafult "args.early_stopping_checkpoints" while weight averaging 
        ## this will be useful while called for .sh script 
        if est_cpts:
                args.early_stopping_checkpoints=est_cpts

        #args=parser.parse_args(namespace=ns)
        args.SWA_random_tag = SWA_random_tag
        ###make SWA name 
        model_name = str(args.model_dir).split('/')[-1]
        ct=model_name+'_SWA_random_tag_'+str(args.SWA_random_tag) + '_args_ealystpping_checkpoints_'+str(args.early_stopping_checkpoints)

        if not isfile(join(args.model_dir,ct)):
                args.gpu=0
                args.pre_trained_weight="0"
                model,optimizer=Initialize_Trans_model(args)
                ##check the Weight averaged file and if the file does not exist then lcreate them
                ## if the file exists load them
                model_names,checkpoint_ter = get_best_weights(args.weight_text_file, args.Res_text_file)
                model_names_checkpoints=model_names[:args.early_stopping_checkpoints]
                swa_files=model_name+'_SWA_random_tag_weight_files_'+str(args.SWA_random_tag) + '_args_ealystpping_checkpoints_'+str(args.early_stopping_checkpoints)
                outfile=join(args.model_dir,swa_files)
                #-----------
                with open(outfile,'a+') as outfile:
                        print(model_names_checkpoints,file=outfile)
                #-----------
                model = Stocasting_Weight_Addition(model, model_names_checkpoints)
                print("Saving_the_SWA_model with Name:",join(args.model_dir,ct))
                torch.save(model.state_dict(),join(args.model_dir,ct))
                model.eval()
        else:
                ## ##load the required weights 
                args.pre_trained_weight = join(args.model_dir,str(ct))
                model, optimizer = Initialize_Trans_model(args)
                model.eval()
        return model,optimizer


# model_dir="/mnt/matylda3/vydana/HOW2_EXP/LIBRISPEECH_RNNLM_V2/models/Trans_lang_6_512_1024_8_10000_accm8"
# SWA_random_tag=1
# print(Load_Transformer_LM_model(model_dir, SWA_random_tag))
