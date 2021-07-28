#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import numpy as np
import fileinput
from numpy.random import permutation
##------------------------------------------------------------------
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
#----------------------------------------
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
os.environ['PYTHONUNBUFFERED'] = '1'
import glob
from statistics import mean
import json
import kaldi_io
#*************************************************************************************************************************
####### Loading the Parser and default arguments
#import pdb;pdb.set_trace()
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
import Attention_arg
from Attention_arg import parser
args = parser.parse_args()

###save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

#####setting the gpus in the gpu cluster
#**********************************
if args.gpu:
        cuda_command = 'nvidia-smi --query-gpu=memory.free,memory.total --format=csv | tail -n+2 | ' \
               'awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
        oooo=subprocess.check_output(cuda_command, shell=True)

        dev_id=str(oooo).lstrip('b').strip("'").split('\\n')[0]
        os.environ["CUDA_VISIBLE_DEVICES"]=dev_id

        gpu_no=os.environ["CUDA_VISIBLE_DEVICES"]
        print("Using gpu number" + str(gpu_no))
        dummy_variable=torch.zeros((10,10))
dummy_variable = dummy_variable.cuda() if args.gpu else dummy_variable
#----------------------------------------------------------------
#=================================================================
#sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Dataloader_for_AM_v1 import DataLoader
from Initializing_model_LSTM_SS_v2_args import Initialize_Att_model
from Load_sp_model import Load_sp_models
from Training_loop import train_val_model
from Spec_Augument import Spec_Aug_freqCont as Spec_Aug
from CMVN import CMVN
from utils__ import weights_init,reduce_learning_rate,read_as_list,gaussian_noise,plotting
from user_defined_losses import preprocess,compute_cer
from Decoding_loop import get_cer_for_beam
#===================================================================
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)

png_dir=args.model_dir+'_png'
if not isdir(png_dir):
        os.makedirs(png_dir)
############################################
#================================================================================
#=======================================================
def main():  
        ##
        ##Load setpiece models for Dataloaders
        Word_model=Load_sp_models(args.Word_model_path)
        Char_model=Load_sp_models(args.Char_model_path)
        ###initilize the model
        model,optimizer=Initialize_Att_model(args)
        #============================================================
        #------------------------------------------------------------  
        train_gen = DataLoader(files=glob.glob(args.train_path + "*"),
                                max_batch_label_len=20000,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=args.max_feat_len,
                                max_label_len=args.max_label_len,
                                Word_model=Word_model,
                                Char_model=Char_model,
                                text_file=args.text_file)    

        dev_gen = DataLoader(files=glob.glob(args.dev_path + "*"),
                                max_batch_label_len=20000,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=args.max_feat_len,
                                max_label_len=args.max_label_len,
                                Word_model=Word_model,
                                Char_model=Char_model,
                                text_file=args.text_file)
        #
        #
        #Flags that may change while training 
        weight_noise_flag=False
        spec_aug_flag=False
        val_history=np.zeros(args.nepochs)   
        #======================================
        for epoch in range(args.nepochs):
            ##start of the epoch
            tr_CER=[]; tr_BPE_CER=[]; L_train_cost=[]
            model.train();
            for trs_no in range(args.validate_interval):

                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"

                Output_trainval_dict=train_val_model(args = args, 
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
                attention_map=Output_trainval_dict.get('attention_record').data.cpu().numpy()
                #==========================================
                if (trs_no%args.tr_disp==0):
                    print("tr ep:==:>",epoch,"sampl no:==:>",trs_no,"train_cost==:>",mean(L_train_cost),"CER:",mean(tr_CER),'BPE_CER',mean(tr_BPE_CER),flush=True)    
                    #------------------------
                    if args.plot_fig_training:
                        plot_name=join(png_dir,'train_epoch'+str(epoch)+'_attention_single_file_'+str(trs_no)+'.png')
                        #print(plot_name)
                        plotting(plot_name,attention_map)
            
            ###validate the model
            #=========================================
            #Vl_Output_dict=validate_the_model(args,epoch,dev_gen,model_encoder,model_decoder,encoder_optim,decoder_optim)
            #=======================================================
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
                Val_Output_trainval_dict=train_val_model(args=args,
                                                        model = model,
                                                        optimizer = optimizer,
                                                        data_dict = B1,
                                                        weight_noise_flag=False,
                                                        spec_aug_flag=False,
                                                        trainflag = False)
            
                L_val_cost.append(Val_Output_trainval_dict.get('cost_cpu'))
                Vl_CER.append(Val_Output_trainval_dict.get('Char_cer'))
                Vl_BPE_CER.append(Val_Output_trainval_dict.get('Word_cer'))
                attention_map=Val_Output_trainval_dict.get('attention_record').data.cpu().numpy()

                #======================================================     
                #======================================================
                if (vl_smp%args.vl_disp==0) or (val_examples==args.max_val_examples-1):
                    print("val epoch:==:>",epoch,"val smp no:==:>",vl_smp,"val_cost:==:>",mean(L_val_cost),"CER:",mean(Vl_CER),'BPE_CER',mean(Vl_BPE_CER),flush=True)    
                    if args.plot_fig_validation:
                        plot_name=join(png_dir,'val_epoch'+str(epoch)+'_attention_single_file_'+str(vl_smp)+'.png')
                        ##print(plot_name)                                    
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
            #######################################################                    
            # decoder_ct="decoder_" + str(ct)
            # print(decoder_ct)
            # torch.save(model_decoder.state_dict(),join(args.model_dir,str(decoder_ct)))

            #######################################################
            ###open the file write and close it to avoid delays
            with open(args.weight_text_file,'a+') as weight_saving_file:
                print(join(args.model_dir,str(ct)), file=weight_saving_file)

            with open(args.Res_text_file,'a+') as Res_saving_file:
                print(float(mean(Vl_CER)), file=Res_saving_file)
            #=================================

            #early_stopping and checkpoint averaging: 
            if args.reduce_learning_rate_flag:
            #=================================================================
                A=val_history
                Non_zero_loss=A[A>0]
                min_cpts=np.argmin(Non_zero_loss)
                Non_zero_len=len(Non_zero_loss)

                if ((Non_zero_len-min_cpts) > 1) and epoch>args.lr_redut_st_th: #args.early_stopping_checkpoints:                                
                    reduce_learning_rate(optimizer)
                    #reduce_learning_rate(decoder_optim)

                    ###start regularization only when model starts to overfit
                    weight_noise_flag=True
                    spec_aug_flag=True
                #------------------------------------
                for param_group in optimizer.param_groups:
                    lr=param_group['lr']                
                    print("learning rate of the epoch:",epoch,"is",lr)   

                if args.early_stopping:
                    #------------------------------------
                    if lr<=1e-8:
                        print("lr reached to a minimum value")
                        exit(0)
            #----------------------------------
            #**************************************************************
            #**************************************************************







# def validate_the_model(arsg,epoch,dev_gen,model_encoder,model_decoder,encoder_optim,decoder_optim):
#                 #=======================================================
#                 model_encoder.eval()
#                 model_decoder.eval()
#                 #=======================================================
#                 Vl_CER=[]; Vl_BPE_CER=[];L_val_cost=[]
#                 val_examples=0
#                 for vl_smp in range(args.max_val_examples):

#                     B1 = dev_gen.next()
#                     smp_feat = B1.get('smp_feat')
#                     val_examples+=smp_feat.shape[0]
#                     assert B1 is not None, "None should never come out of the DataLoader"

#                     ##brak when the examples are more
#                     if (val_examples >= args.max_val_examples):
#                         break;
#                     #--------------------------------------                
#                     Val_Output_trainval_dict=train_val_model(args=args,
#                                                             model_encoder = model_encoder,
#                                                             model_decoder = model_decoder,
#                                                             encoder_optim = encoder_optim,
#                                                             decoder_optim = decoder_optim,
#                                                             data_dict = B1,
#                                                             weight_noise_flag=False,
#                                                             spec_aug_flag=False,
#                                                             trainflag = False)
                
#                     L_val_cost.append(Val_Output_trainval_dict.get('cost_cpu'))
#                     Vl_CER.append(Val_Output_trainval_dict.get('Char_cer'))
#                     Vl_BPE_CER.append(Val_Output_trainval_dict.get('Word_cer'))
#                     attention_map=Val_Output_trainval_dict.get('attention_record').data.cpu().numpy()

#                     #======================================================     
#                     #======================================================
#                     if (vl_smp%args.vl_disp==0) or (val_examples==args.max_val_examples-1):
#                         print("val epoch:==:>",epoch,"val smp no:==:>",vl_smp,"val_cost:==:>",mean(L_val_cost),"CER:",mean(Vl_CER),'BPE_CER',mean(Vl_BPE_CER),flush=True)    
#                         if args.plot_fig_validation:
#                             plot_name=join(png_dir,'val_epoch'+str(epoch)+'_attention_single_file_'+str(vl_smp)+'.png')
#                             ##print(plot_name)                                    
#                             plotting(plot_name,attention_map)                            
#                         #----------------------------------------------------
#                         #====================================================

#                 Vl_Output_dict={'L_val_cost':L_val_cost, 
#                                 'Vl_CER':Vl_CER,
#                                 'Vl_BPE_CER':Vl_BPE_CER }
#                 return Vl_Output_dict 

#=============================================================================================
#=============================================================================================
if __name__ == '__main__':
    main()



