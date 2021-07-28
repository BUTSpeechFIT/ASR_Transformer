#!/usr/bin/python
import kaldi_io
import sys
import os
from os.path import join, isdir
from numpy.random import permutation
import itertools
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import queue
from threading  import Thread
import random
import glob


import sys
sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
import CMVN
from CMVN import CMVN
from Load_sp_model import Load_sp_models


#===============================================
#-----------------------------------------------  
class DataLoader(object):

    def __init__(self,files, max_batch_label_len, max_batch_len, max_feat_len, max_label_len, Word_model, Char_model, queue_size=100,apply_cmvn=1):

        self.files = files
        if self.files==[]:
                print('input to data generator in empty')
                exit(0)


        self.text_file_dict ={} 

        self.Word_model = Word_model
        self.Char_model = Char_model
        self.max_batch_len = max_batch_len
        self.max_batch_label_len = max_batch_label_len
        self.max_feat_len = max_feat_len
        self.max_label_len = max_label_len
        self.apply_cmvn = apply_cmvn


        self.queue = queue.Queue(queue_size)
        self.Word_padding_id = self.Word_model.__len__()
        self.Char_padding_id = self.Char_model.__len__()
        self.word_space_token   = self.Word_model.EncodeAsIds('_____')[0]
        
        self.datareader=DataLoader.process_everything_in_parllel_dummy2


        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()

    
    def __reset_the_data_holders(self):
        self.batch_data=[]
        self.batch_labels=[]
        self.batch_names=[]
        self.batch_length=[]
        self.batch_label_length=[]
        
        self.batch_word_labels=[]
        self.batch_word_label_length=[]
        
        self.batch_word_text=[]
        self.batch_word_text_length=[]

        self.batch_word_text_tgt=[]
        self.batch_word_text_length_tgt=[]
    
    #---------------------------------------------------------------------
    def make_batching_dict(self):
        #----------------------------------------
        smp_feat=pad_sequences(self.batch_data,maxlen=max(self.batch_length),dtype='float32',padding='post',value=0.0)
        smp_char_labels=pad_sequences(self.batch_labels,maxlen=max(self.batch_label_length),dtype='int32',padding='post',value=self.Char_padding_id) 
        smp_word_label=pad_sequences(self.batch_word_labels,maxlen=max(self.batch_word_label_length),dtype='int32',padding='post',value=self.Word_padding_id)
        smp_trans_text=pad_sequences(self.batch_word_text, maxlen=max(self.batch_word_text_length),dtype=object,padding='post',value=' ')
        smp_trans_text_tgt=pad_sequences(self.batch_word_text_tgt, maxlen=max(self.batch_word_text_length_tgt),dtype=object,padding='post',value=' ')

        batch_data_dict={
            'smp_names':self.batch_names,
            'smp_feat':smp_feat,
            'smp_char_label':smp_char_labels,
            'smp_word_label':smp_word_label,
            'smp_trans_text':smp_trans_text,
            'smp_trans_text_tgt': smp_trans_text_tgt,
            'smp_feat_length':self.batch_length,
            'smp_label_length':self.batch_label_length,
            'smp_word_label_length':self.batch_word_label_length,
            'smp_word_text_length':self.batch_word_text_length,
            'smp_word_text_length_tgt':self.batch_word_text_length_tgt}
        return batch_data_dict
    #------------------------------------------
    #------------------------------------------
    def __load_data(self):
        ###initilize the lists
        while True:
            self.__reset_the_data_holders()
            max_batch_label_len = self.max_batch_label_len
            random.shuffle(self.files)
            for inp_file in self.files:
                print(inp_file)
                with open(inp_file) as f:
                        ###########################################################
                        ###########################################################
                        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                                results=[executor.submit(self.datareader,line) for line in f]
                                for R in concurrent.futures.as_completed(results):

                                    dataread_dict=R.result()

                                    if dataread_dict==None:
                                        continue;
                                   
                                    #--------------------------
                                    key = dataread_dict['key'];
                                    #print('************************',key)
                                    mat = dataread_dict['mat'];
                                    
                                    word_tokens = dataread_dict['Src_tokens'];
                                    
                                    word_labels = dataread_dict['Src_Words_Text'];
                                    
                                    char_tokens = dataread_dict['Tgt_tokens'];
                
                                    char_labels = dataread_dict['Tgt_Words_Text'];
        
                                    scp_path = dataread_dict['scp_path'];
                                    
                                    dataread_dict={}
                                    R._result = None


                                    if (scp_path!='None'):
                                        ##ASR data-----
                                        if self.apply_cmvn:
                                            mat = CMVN(mat)

                                    #----------------------------------------------------------------------------------------------------------------
                                    if (mat.shape[0]>self.max_feat_len) or (mat.shape[0]<len(word_tokens)) or (len(word_tokens) > self.max_label_len):
                                            
                                            continue;

                                    #==============================================================
                                    ###Add to the list
                                    ####
                                    self.batch_data.append(mat)                
                                    self.batch_names.append(key)
                                    self.batch_length.append(mat.shape[0])

                                    self.batch_labels.append(char_tokens)
                                    self.batch_label_length.append(len(char_tokens))
                                    
                                    self.batch_word_labels.append(word_tokens)
                                    self.batch_word_label_length.append(len(word_tokens))

                                    self.batch_word_text.append(char_labels)
                                    self.batch_word_text_length.append(len(char_labels))

                                    self.batch_word_text_tgt.append(word_labels)
                                    self.batch_word_text_length_tgt.append(len(word_labels))   
                                    #==============================================================

                                    #==============================================================
                                    # total_labels_in_batch is used to keep track of the length of sequences in a batch, just make sure it does not overflow the gpu
                                    ##in general lstm training we are not using this because self.max_batch_len will be around 10-20 and self.max_batch_label_len is usuvally set very high                         
                                    expect_len_of_features=max(max(self.batch_length,default=0),mat.shape[0])
                                    expect_len_of_labels=max(max(self.batch_label_length,default=0),len(char_tokens))

                                    total_labels_in_batch= (expect_len_of_features + expect_len_of_labels)*(len(self.batch_names)+4)

                                    ###check if ypu have enough labels output and if you have then push to the queue
                                    ###else keep adding them to the lists
                                    if total_labels_in_batch > self.max_batch_label_len or len(self.batch_data)==self.max_batch_len:
                                                # #==============================================================
                                                # ####to clumsy -------> for secound level of randomization 
                                                # CCCC=list(zip(batch_data,batch_names,batch_labels,batch_word_labels,batch_word_text,batch_label_length,batch_length,batch_word_label_length,batch_word_text_length))
                                                # random.shuffle(CCCC)
                                                # batch_data,batch_names,batch_labels,batch_word_labels,batch_word_text,batch_label_length,batch_length,batch_word_label_length,batch_word_text_length=zip(*CCCC)
                                                # #==============================================================

                                                batch_data_dict = self.make_batching_dict()
                                                self.queue.put(batch_data_dict)
                                                ###after pushing data to lists reset them
                                                self.__reset_the_data_holders()
            

            # if len(self.batch_names)>0:
            #     ### Collect the left over stuff  as the last batch
            #     #-----------------------------------------------
            #     batch_data_dict = self.make_batching_dict()
            #     self.queue.put(batch_data_dict)

    def next(self, timeout=30):
        return self.queue.get(block=True, timeout=timeout)

    #==================================================================================================================================
    @staticmethod 
    def process_everything_in_parllel_dummy2(line):
        #output_dict={'key':None,'mat':None,'Src_tokens':None,'Src_Words_Text':None,'Tgt_tokens':None,'Tgt_Words_Text':None,'scp_path':None}
        #============================
        split_lines=line.split(' @@@@ ')
        #============================
        key = split_lines[0]
        scp_path = split_lines[1] #will be 'None' fo MT setup
        scp_path = 'None' if scp_path == '' else scp_path
        #print('-------',key)
        #============================
        ### Char labels
        #============================
        src_text = split_lines[3] 
        src_tok = split_lines[4] 

        tgt_text = split_lines[5]
        tgt_tok = split_lines[6]

        # Src_tokens = src_tok
        # Tgt_tokens = tgt_tok

        if tgt_text=='None':
            ###ASR Data
            Tgt_tokens=list(map(lambda x: x*0, tgt_tok))

        if ('None' not in src_tok) and ('None' not in tgt_tok) and (len(src_tok)>0) and (len(tgt_tok)>0):

            Src_tokens = [int(i) for i in src_tok.split(' ')]
            Tgt_tokens = [int(i) for i in tgt_tok.split(' ')]
            #============================
            ### text 
            #============================
            Src_Words_Text = src_text.split(' ')
            Tgt_Words_Text = tgt_text.split(' ')
            #--------------------------                                
            #--------------------------
            if not (scp_path == 'None'):
                mat = kaldi_io.read_mat(scp_path)
            else:
                mat=np.zeros((100,83),dtype=np.float32)

            ###########################################
            output_dict={'key':key,
                        'mat':mat,
                        'Src_tokens':Src_tokens,
                        'Src_Words_Text':Src_Words_Text,
                        'Tgt_tokens':Tgt_tokens,
                        'Tgt_Words_Text':Tgt_Words_Text,
                        'scp_path':scp_path}
        else:
            return None

        return output_dict
#===================================================================


# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/KAT_Attention')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)


# ###debugger
# args.Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.text_file = '/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# args.train_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/'
# args.dev_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/'
# Word_model=Load_sp_models(args.Word_model_path)
# Char_model=Load_sp_models(args.Char_model_path)
# train_gen = DataLoader(files=glob.glob(args.train_path + "*"),max_batch_label_len=20000, max_batch_len=4,max_feat_len=2000,max_label_len=200,Word_model=Word_model,Char_model=Char_model,text_file=args.text_file)
# for i in range(10):
#     B1 = train_gen.next()
#     print(B1.keys())
#     #breakpoint()

