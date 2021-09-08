#!/usr/bin/python3

# authors: Hari Vydana, Santosh Kesiraju
#

import argparse
import sys
import os
from os.path import join, isdir

import sentencepiece as spm
sp = spm.SentencePieceProcessor()

#s.Load('spm.model')

parser = argparse.ArgumentParser()
parser.add_argument("text_file", help="path to plain text file to train the tokenizers")
parser.add_argument("model_dir_prefix", help="path to save tokenizer models")
parser.add_argument("tokenizer_type", choices=["char", "word", "bpe", "unigram"], help="choices of tokenizer")
parser.add_argument("vocab_size", type=int, help="max vocab size")
parser.add_argument("-user_defined_tokens", type=str, default="<HES>,<UNK>,<BOS>,<EOS>,{LIPSMACK},{BREATH},{LAUGH},{COUGH}",
                    help="user defined tokens")
args = parser.parse_args()

utt_text_normalized=args.text_file
model_prefix=args.model_dir_prefix
tokenizer_type = args.tokenizer_type
vocab_size=args.vocab_size

user_generated_strings=args.user_defined_tokens
print("Pre-defined strings:", user_generated_strings.split(','))

vocab_size = int(vocab_size)+len(user_generated_strings.split(','))
vocab_size = int(vocab_size+2)


#import pdb;pdb.set_trace()

#Special_tokens=str(sys.argv[4])
#user_generated_strings+=Special_tokens

#LIPSMACK', 'BREATH', 'LAUGH','COUGH','%UH','%UM','%AH','%EH','%ER','%UH','UH','UM','AH','EH','ER','{LIPSMACK}', '{BREATH}', '{LAUGH}','{COUGH}','<UNK>','<unk>','-','(<HES>)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
spm_str="--input="+str(utt_text_normalized)+f" --model_prefix={model_prefix}_{tokenizer_type} --vocab_size={vocab_size} --model_type={tokenizer_type}" + "  --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols="+str(user_generated_strings)

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# bpe_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_bpe'+' --vocab_size='+str(vocab_size)+' --model_type=bpe --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings)
# #--------------------------------------------------------------------------------------------------------------------------------------------------------
# word_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_word'+' --vocab_size='+str(vocab_size)+' --model_type=word --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings)
# #----------------------------------------------------------------------------------------------------------------------------------------------------------
# char_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_char'+' --vocab_size='+str(vocab_size)+' --model_type=char --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings)

#char_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_char'+' --vocab_size='+str(vocab_size)+' --model_type=char --character_coverage=1.0 --pad_id=500 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[blk] --unk_piece=[HES]  --bos_piece=[UNK]'
#--------------------------------------------------------------------------------------------------------------------------------------------------------

print(spm_str)

spm.SentencePieceTrainer.Train(spm_str)

# spm.SentencePieceTrainer.Train(bpe_str)

# spm.SentencePieceTrainer.Train(word_str)

# spm.SentencePieceTrainer.Train(char_str)
