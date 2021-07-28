#! /usr/bin/python

import sys
import os
from os.path import join


sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
from Load_sp_model import Load_sp_models
text_dlim=' @@@@ '
#=================================================================
def Search_for_utt(query, search_file,SPmodel):
        #
        while True and query:
                line = search_file.readline()
                line = line.strip()
                splitlines = line.split(' ')
                uttid = splitlines[0]
                utt_text = " ".join(splitlines[1:])

                if query==uttid:
                       if SPmodel:
                               tokens_utt_text = SPmodel.EncodeAsIds(utt_text) 
                               tokens_utt_text = [str(intg) for intg in tokens_utt_text]
                               tokens_utt_text = " ".join(tokens_utt_text)
                               utt_text = utt_text + text_dlim + tokens_utt_text + text_dlim                            
                       else:
                               tokens_utt_text = 'None'
                               utt_text = utt_text + text_dlim + tokens_utt_text + text_dlim 

                       return utt_text

                if not line:
                        print('the uttid '+ uttid +' line not present in Translations')
                        utt_text = 'None'
                        tokens_utt_text = 'None'
                        utt_text = utt_text + text_dlim + tokens_utt_text + text_dlim
                        return utt_text
#================================================================


#output_file='Timit_text_like_MT'
#scp_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/sorted_feats_pdnn_train_scp'
#transcript='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
#Translation='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text_2'

#Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
#Word_model = Load_sp_models(Word_model_path)
#outfile=open(output_file,'w')
#F=open(scp_file,'r')
#count=0


def format_tokenize_data(scp_file,transcript,Translation,outfile,Word_model,Char_model): 
  for F in scp_file:
    F1=open(F,'r')
    while True:
          line=F1.readline()
          if not line:
                  print('finished iterating the file')
                  break;

          line=line.strip()
          split_lines=line.split(' ')

          uttid=split_lines[0]
          utt_text=" ".join(split_lines[1:])
          inp_seq=uttid + text_dlim
          ###scp_file
          #-------------------                
          inp_seq += Search_for_utt(query=uttid, search_file=open(F,'r'),SPmodel=None)
          ###transcriptions
          #-------------------
          inp_seq += Search_for_utt(query=uttid, search_file=open(transcript,'r'),SPmodel=Word_model)
          ###translations
          #-------------------
          inp_seq += Search_for_utt(query=uttid, search_file=open(Translation,'r'),SPmodel=Char_model)
          #-------------------
          print(inp_seq,file=outfile) 
#============================================================================
#format_tokenize_data(scp_file,transcript,Translation,outfile)
      
















