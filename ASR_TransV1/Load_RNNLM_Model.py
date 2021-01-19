#!/usr/bin/python
import sys
import os
from os.path import join, isdir
#----------------------------------------
import glob
import json
from argparse import Namespace
#**********
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
#-----------------------------------
import torch
##model class
#from RNNLM import RNNLM
from Initializing_RNNLM_model_args import Initialize_RNNLM_model
##config file for RNLM
import RNNLM_config
from RNNLM_config import parser

def Load_RNNLM_model(RNNLM_model_weight_path):
        ####save architecture for decoding
        RNNLM_path="/".join(RNNLM_model_weight_path.split('/')[:-1])
        RNNLM_model_path_name=join(RNNLM_path,'model_architecture_')
        print("Using the language model in the path", RNNLM_model_path_name)
        with open(RNNLM_model_path_name, 'r') as f:
            RNNLM_TEMP_args = json.load(f)
        RNNLM_ns = Namespace(**RNNLM_TEMP_args)
        
        #RNNLM=parser.parse_args(namespace=RNNLM_ns)
        ##==================================
        RNNLM_ns.gpu=0
        RNNLM_ns.pre_trained_weight=RNNLM_model_weight_path
        LM_model,optimizer=Initialize_RNNLM_model(RNNLM_ns)
        return LM_model,optimizer 
