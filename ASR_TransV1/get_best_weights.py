#!/usr/bin/python
import sys
import os
from os.path import join, isdir, isfile
#----------------------------------------
import glob
import json

#**********
from utils__ import plotting,read_as_list
#--------------------------------
def get_best_weights(weight_text_file,Res_text_file):
        weight_list=read_as_list(weight_text_file)
        ERROR_list=read_as_list(weight_text_file+'_Res')
        weight_acc_dict = dict(zip(ERROR_list,weight_list))
       
        sorted_weight_acc_dict = sorted(weight_acc_dict.items(), key=lambda x: x[0],reverse=False)
        check_points_list = sorted_weight_acc_dict

       
        model_names=[W[1] for W in check_points_list]
        checkpoint_ter=[W[0] for W in check_points_list]
        round_checkpoint_ter=[str(round(float(N),2)) for N in checkpoint_ter]
        #print('checkpoint_TER,checkpoint_names',round_checkpoint_ter,model_names)
        return model_names,checkpoint_ter
#--------------------------------
