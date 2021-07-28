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

#import safe_gpu
#from safe_gpu import safe_gpu
#gpu_owner = safe_gpu.GPUOwner()
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
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
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

#Dataloader_for_AM_v3_dev.py
from Dataloader_for_AM_v3_dev import DataLoader
from utils__ import weights_init,reduce_learning_rate,read_as_list,gaussian_noise,plotting
#==============================================================
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
from TRANSFORMER_ASR_V1 import Transformer
from Initializing_Transformer_ASR_CTC import Initialize_Att_model
from Transformer_Training_loop import train_val_model
#from Transformer_Training_loop_PS import train_val_model
from Load_sp_model import Load_sp_models
##==================================
#==============================================================
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)

png_dir=args.model_dir+'_png'
if not isdir(png_dir):
        os.makedirs(png_dir)
############################################
#=============================================================
def main():
        ##Load setpiece models for Dataloaders
        Word_model=Load_sp_models(args.Word_model_path)
        Char_model=Load_sp_models(args.Char_model_path)
        ###initilize the model
        model,optimizer=Initialize_Att_model(args)
        #============================================================
        #------------------------------------------------------------  
        #
        train_gen = DataLoader(files=glob.glob(args.data_dir + "train_splits/*"), 
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


        #Flags that may change while training 
        if args.spec_aug_flag==2:
                weight_noise_flag=False
                spec_aug_flag=True
        else:
              weight_noise_flag=False
              spec_aug_flag=False
        val_history=np.zeros(args.nepochs)
        #======================================
        for epoch in range(args.nepochs):
            ##start of the epoch
            tr_CER=[]; tr_BPE_CER=[]; L_train_cost=[]
            model.train();
            validate_interval = int(args.validate_interval * args.accm_grad) if args.accm_grad>0 else args.validate_interval
            for trs_no in range(validate_interval):
                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"

                

                print(B1)
                #exit(0)

                

                Output_trainval_dict=train_val_model(smp_no=trs_no,
                                                    args = args, 
                                                    model = model,
                                                    optimizer = optimizer,
                                                    data_dict = B1,
                                                    weight_noise_flag=weight_noise_flag,
                                                    spec_aug_flag=spec_aug_flag,
                                                    trainflag = True)
                #
                #
                #get the losses form the dict
                L_train_cost.append(Output_trainval_dict.get('cost_cpu'))
                tr_CER.append(Output_trainval_dict.get('Char_cer'))
                tr_BPE_CER.append(Output_trainval_dict.get('Word_cer'))
                #attention_map=Output_trainval_dict.get('attention_record').data.cpu().numpy()
                print(L_train_cost,tr_CER,tr_BPE_CER)
                exit(0)
                #==========================================
                if (trs_no%args.tr_disp==0):
                    print("tr ep:==:>",epoch,"sampl no:==:>",trs_no,"train_cost==:>",mean(L_train_cost),"CER:",mean(tr_CER),'BPE_CER',mean(tr_BPE_CER),flush=True)    
                    #------------------------
                    if args.plot_fig_training:
                        plot_name=join(png_dir,'train_epoch'+str(epoch)+'_attention_single_file_'+str(trs_no)+'.png')

                        plotting(plot_name,attention_map)
                #exit(0)            

            ###validate the model
            model.eval()
            #=======================================================
            Vl_CER=[]; Vl_BPE_CER=[];L_val_cost=[]
            val_examples=0
            for vl_smp in range(args.max_val_examples):
                B1 = dev_gen.next()
                smp_feat = B1.get('smp_feat')
                val_examples+=smp_feat.shape[0]
                assert B1 is not None, "None should never come out of the DataLoader"

                ##brak when the examples are more
                if (val_examples >= args.max_val_examples):
                    break;
                #--------------------------------------                
                Val_Output_trainval_dict=train_val_model(smp_no=trs_no,
                                                        args=args,
                                                        model = model,
                                                        optimizer = optimizer,
                                                        data_dict = B1,
                                                        weight_noise_flag=False,
                                                        spec_aug_flag=False,
                                                        trainflag = False)
            
                L_val_cost.append(Val_Output_trainval_dict.get('cost_cpu'))
                Vl_CER.append(Val_Output_trainval_dict.get('Char_cer'))
                Vl_BPE_CER.append(Val_Output_trainval_dict.get('Word_cer'))
                #attention_map=Val_Output_trainval_dict.get('attention_record').data.cpu().numpy()

                #======================================================     
                #======================================================
                if (vl_smp%args.vl_disp==0) or (val_examples==args.max_val_examples-1):
                    print("val epoch:==:>",epoch,"val smp no:==:>",vl_smp,"val_cost:==:>",mean(L_val_cost),"CER:",mean(Vl_CER),'BPE_CER',mean(Vl_BPE_CER),flush=True)    

                    if args.plot_fig_validation:
                        plot_name=join(png_dir,'val_epoch'+str(epoch)+'_attention_single_file_'+str(vl_smp)+'.png')                                 
                        plotting(plot_name,attention_map)                             
            #----------------------------------------------------
#==================================================================
            val_history[epoch]=(mean(Vl_CER)*100)
            print("val_history:",val_history[:epoch+1])
            #================================================================== 
            ####saving_weights 
            ct="model_epoch_"+str(epoch)+"_sample_"+str(trs_no)+"_"+str(mean(L_train_cost))+"___"+str(mean(L_val_cost))+"__"+str(mean(Vl_CER))
            print(ct)
            torch.save(model.state_dict(),join(args.model_dir,str(ct)))
            ####saving otpimizer helped Transformer
            #torch.save(optimizer.state_dict(),join(args.model_dir,str(ct)+'_opt'))

            #######################################################                    
            #######################################################
            ###open the file write and close it to avoid delays
            with open(args.weight_text_file,'a+') as weight_saving_file:
                print(join(args.model_dir,str(ct)), file=weight_saving_file)

            with open(args.Res_text_file,'a+') as Res_saving_file:
                print(float(mean(Vl_CER)), file=Res_saving_file)
            #=================================
             # early_stopping and checkpoint averaging:                    
            if args.early_stopping:
                 A=val_history
                 Non_zero_loss=A[A>0]
                 min_cpts=np.argmin(Non_zero_loss)
                 Non_zero_len=len(Non_zero_loss)

                 if ((Non_zero_len-min_cpts)>1):
                                weight_noise_flag=True
                                spec_aug_flag=True

                 if (Non_zero_len-min_cpts) > args.early_stopping_patience:                                
                    print("The model is early stopping........","minimum value of model is:",min_cpts)
                    exit(0)

#=======================================================
#=============================================================================================
if __name__ == '__main__':
    main()



