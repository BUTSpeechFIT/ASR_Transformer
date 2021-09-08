#!/usr/bin/bash

if [ $# -ne 4 ]; then
    echo -e "\n+ usage: $0 <path to all text> <dataset_name> <tokenizer_type> <num_of_tokens>"
    echo -e "  where:"
    echo -e "     - all text is the concatenation of train + val text segments\
of the dataset you are working with"
    echo -e "     - dataset_name is the simple name of the dataset you are working with"
    echo -e "     - tokenizer_type: char, word, bpe, unigram"
    echo -e "     - num_of_tokens (INT) is the number of subword (BPE) tokens"
    echo -e "\n\n  Dependencies: utt_piece_training_nonorm.py\n\n"
    exit;
fi

#cat train_text dev_text test_text > All_text


text_file=$1
db_name=$2
tokenizer_type=$3
no_of_tokens=$4

model_dir="models/${db_name}_${no_of_tokens}"
mkdir -p $model_dir
model_prefix="${model_dir}/${db_name}_${no_of_tokens}"

name_suffix='.txt'

#-----------
cut -d " " -f1 $text_file > ${db_name}.utt_id
#-----------
cut -d " " -f2- ${text_file} | sed 's/  */ /g' |  sort -u > ${db_name}.utt_text
#-----------

#no_of_tokens=`cat utt_text|tr -s " " "\n"|sort|uniq|wc -l`
#echo "$no_of_tokens"
# echo "$text_file"

python3 utt_piece_training_nonorm.py ${db_name}.utt_text $model_prefix ${tokenizer_type} $no_of_tokens
