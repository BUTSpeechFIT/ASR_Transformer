#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
from Spec_Augument import Spec_Aug_freqCont as Spec_Aug
from utils__ import weights_init,gaussian_noise

#---------------------------------------
def train_val_model(**kwargs):
        smp_no=kwargs.get('smp_no')
        args = kwargs.get('args')
        model = kwargs.get('model')
        optimizer= kwargs.get('optimizer')
 
        trainflag = kwargs.get('trainflag')
        weight_noise_flag = kwargs.get('weight_noise_flag')
        spec_aug_flag = kwargs.get('spec_aug_flag')

        B1 = kwargs.get('data_dict')
        smp_feat = B1.get('smp_feat')
        smp_char_label = B1.get('smp_char_label')
        smp_word_label = B1.get('smp_word_label')
        smp_trans_text = B1.get('smp_trans_text')  

        #################finished expanding the keyword arguments#########
        ##===========================================
        if trainflag and args.spec_aug_flag and spec_aug_flag:
               smp_feat_mask = Spec_Aug(smp_feat,args.min_F_bands,args.max_F_bands,args.time_drop_max,args.time_window_max)
               smp_feat = smp_feat * smp_feat_mask

        # #==========================================
        if trainflag and (args.weight_noise_flag) and weight_noise_flag:
                 params = list(model.parameters()) 
                 param = [gaussian_noise(param, args.gpu) for param in params]
        #============================================
OM=False
        ###################################################################
        input=torch.from_numpy(smp_feat).float()
        Char_target=torch.LongTensor(smp_char_label)
        Word_target=torch.LongTensor(smp_word_label)
        #-----------------------------------------------------------------
        input=input.cuda() if args.gpu else input
        Word_target=Word_target.cuda() if args.gpu else Word_target

        #--------------------------------
        OOM=False
        if trainflag:
            try:
                Decoder_out_dict = model(input,Word_target)                              
                #break;
            except Exception as e:
                   if 'CUDA out of memory' in str(e):
                      OOM=True
                      torch.cuda.empty_cache()
                      print("The model in OOM condition","smp_no",smp_no,"batch size for the batch is:",input.shape)
                      #break;
            if OOM:
                  batch_size=input.shape[0]
                  input=input[:2]
                  Word_target=Word_target[:2]
                  print("The model running under OOM condition","smp_no",smp_no,"batch size for the batch is:",2)
                  Decoder_out_dict = model(input,Word_target)
        else:
            with torch.no_grad():
                    Decoder_out_dict = model(input,Word_target)
        #--------------------------------
        cost=Decoder_out_dict.get('cost')

        ###training with accumilating gradients
        if trainflag:
                #Done always before .backword()
                cost=cost/args.accm_grad
                cost.backward()
                if args.clip_grad_norm != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
                #
                
                cost.detach()   

                ###gradient accumilation
                if(smp_no%args.accm_grad)==0:
                    optimizer.step()
                    optimizer.zero_grad()
                cost_cpu=cost.item()
        #--------------------------------------
        cost_cpu = cost.item() 

        ###output a dict
        #==================================================    
        Output_trainval_dict={
                            'cost_cpu':cost_cpu,
                            'dec_slf_attn_list':Decoder_out_dict.get('dec_slf_attn_list'),
                            'dec_enc_attn_list':Decoder_out_dict.get('dec_enc_attn_list'),
                            'Char_cer':Decoder_out_dict.get('Char_cer'),
                            'Word_cer':Decoder_out_dict.get('Word_cer')}
        return Output_trainval_dict
#=========================================================




