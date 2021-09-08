#! /usr/bin/python

# *******************************
import sys
import os
from os.path import join, isdir
from random import shuffle
import glob


# scp_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/sorted_feats_pdnn_train_scp'
# transcript='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# Translation='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'

# Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'

# Word_model = Load_sp_models(Word_model_path)
# Char_model=Load_sp_models(Char_model_path)


# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1')
from Load_sp_model import Load_sp_models
from Make_ASR_scp_text_format_fast import format_tokenize_data


import Transformer_arg
from Transformer_arg import parser

args = parser.parse_args()


if not isdir(args.data_dir):
    os.makedirs(args.data_dir)


format_tokenize_data(
    scp_file=glob.glob(args.train_path + "*"),
    transcript=args.text_file,
    Translation=args.text_file,
    outfile=open(join(args.data_dir, "train_scp"), "w"),
    Word_model=args.Word_model_path,
    Char_model=args.Char_model_path,
)
format_tokenize_data(
    scp_file=glob.glob(args.dev_path + "*"),
    transcript=args.text_file,
    Translation=args.text_file,
    outfile=open(join(args.data_dir, "dev_scp"), "w"),
    Word_model=args.Word_model_path,
    Char_model=args.Char_model_path,
)
