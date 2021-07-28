#!/usr/bin/python
from os.path import join, isdir
import numpy as np
##------------------------------------------------------------------
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
#----------------------------------------
import glob
from statistics import mean
import json
import kaldi_io

from CMVN import CMVN
from utils__ import plotting
from user_defined_losses import compute_cer

from Load_sp_model import Load_sp_models


#from Decoding_loop import get_cer_for_beam

#=========================================================================================================================================
def get_cer_for_beam(scp_paths_decoding,model,text_file_dict,plot_path_name,args):
    #-----------------------------------
    """ If you see best-hypothesis having worse WER that the remainig beam them tweak with the beam hyperpearmaeters Am_wt, len_pen, gamma 
     	If you see best-hypothesis having better performance than the oothers in the beam then improve the model training
    """
    #-----------------------------------
    #import pdb;pdb.set_trace()
    for line in scp_paths_decoding:
        line=line.strip()
        key=line.split(' ')[0]
        
        feat_path=line.split(' ')[1:]
        feat_path=feat_path[0].strip()

        #-----------------------------------
        ####get the model predictions
        Output_seq = model.predict(feat_path,args)
        #Output_seq = model.predict(input,args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen)


        ###get the true label if it exists
        True_label=text_file_dict.get(key,None)
        #-----------------------------------

        #breakpoint()

        llr=[item.get('score').unsqueeze(0) for item in Output_seq]
        norm_llr=torch.nn.functional.softmax(torch.cat(llr,dim=0),dim=0)

        print("final_ouputs",'====','key','Text_seq','LLR','Beam_norm_llr','Yseq','CER')
        print("True_label",True_label)

        #-----------------------------------
        #-----------------------------------
        #import pdb;pdb.set_trace()
        for ind, seq in enumerate(Output_seq):
            Text_seq=seq['Text_seq'][0]
            Text_seq_formatted=[x for x in Text_seq.split(' ') if x.strip()]
            Yseq=seq['yseq'].data.numpy()
            Ynorm_llr=norm_llr[ind].data.numpy()
            Yllr=seq['score'].data.data.numpy()

            #
            #---------------------------------------------
            attention_record=seq.get('alpha_i_list','None')

            #if (attention_record) or (attention_record=='None'):

            if (torch.is_tensor(attention_record)):
                    #---------------------------------------------
                    attention_record=attention_record[:,:,0].transpose(0,1)
                    attention_record = attention_record.data.cpu().numpy()

                    #---------------------------------------------
                    if args.plot_decoding_pics:
                            pname=str(key) +'_beam_'+str(ind)
                            plotting_name=join(plot_path_name,pname)
                            plotting(plotting_name,attention_record)
            
            #-----------------------------------
            #import pdb;pdb.set_trace()
            #breakpoint()
            #-----------------------------------
            if True_label:
                    if Text_seq_formatted==[]:
                       Text_seq_formatted.append('<UNK>')

                    CER=compute_cer(" ".join(Text_seq_formatted)," ".join(True_label),'doesnot_matter')*100
            else:
                    CER=None




            #---------------------------------------------
            if ind==0:
                    print("nbest_output",'=',key,'='," ".join(Text_seq_formatted),'='," ".join(True_label),'=',CER)

            print("final_ouputs",'=',ind,'=',key,'=',Text_seq,'=',Yllr,'=',Ynorm_llr,'=',Yseq,'=',CER)
            #---------------------------------------------

#=========================================================================================================================================
def get_Bleu_for_beam(scp_paths_decoding,model,text_file_dict,plot_path_name,args):
    #-----------------------------------
    """ If you see best-hypothesis having worse WER that the remainig beam them tweak with the beam hyperpearmaeters Am_wt, len_pen, gamma 
        If you see best-hypothesis having better performance than the oothers in the beam then improve the model training
    """
    #-----------------------------------
    #import pdb;pdb.set_trace()
    for line in scp_paths_decoding:
        line=line.strip()
        key=line.split(' ')[0]
        
        feat_path=line.split(' ')[1:]
        feat_path=feat_path[0].strip()

        #-----------------------------------
        ####get the model predictions
        Output_seq = model.predict(feat_path,args)
        #Output_seq = model.predict(input,args.LM_model,args.Am_weight,args.beam,args.gamma,args.len_pen)


        ###get the true label if it exists
        True_label=text_file_dict.get(key,None)
        #-----------------------------------

        
        llr=[item.get('score').unsqueeze(0) for item in Output_seq]
        norm_llr=torch.nn.functional.softmax(torch.cat(llr,dim=0),dim=0)

        print("final_ouputs",'====','key','Text_seq','LLR','Beam_norm_llr','Yseq','CER')
        print("True_label",True_label)

        #-----------------------------------
        #-----------------------------------
        #import pdb;pdb.set_trace()
        for ind, seq in enumerate(Output_seq):
            Text_seq=seq['Text_seq'][0]
            Text_seq_formatted=[x for x in Text_seq.split(' ') if x.strip()]
            Yseq=seq['yseq'].data.numpy()
            Ynorm_llr=norm_llr[ind].data.numpy()
            Yllr=seq['score'].data.data.numpy()

            #
            #---------------------------------------------
            attention_record=seq.get('alpha_i_list','None')

            #if (attention_record) or (attention_record=='None'):

            if (torch.is_tensor(attention_record)):
                    #---------------------------------------------
                    attention_record=attention_record[:,:,0].transpose(0,1)
                    attention_record = attention_record.data.cpu().numpy()

                    #---------------------------------------------
                    if args.plot_decoding_pics:
                            pname=str(key) +'_beam_'+str(ind)
                            plotting_name=join(plot_path_name,pname)
                            plotting(plotting_name,attention_record)
            
            #-----------------------------------
            #-----------------------------------
            if True_label:
                    CER=compute_cer(" ".join(Text_seq_formatted)," ".join(True_label),'doesnot_matter')*100
            else:
                    CER=None

            #---------------------------------------------





            if ind==0:
                    print("nbest_output",'=',key,'='," ".join(Text_seq_formatted),'='," ".join(True_label),'=',CER)

            print("final_ouputs",'=',ind,'=',key,'=',Text_seq,'=',Yllr,'=',Ynorm_llr,'=',Yseq,'=',CER)
            #---------------------------------------------
