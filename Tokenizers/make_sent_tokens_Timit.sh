#!/usr/bin/bash

if [ $# -ne 1 ]; then
    echo "$0 <path to all text>"
    echo "  where all text is the concatenation of train + val text segments\
of the dataset you are working with"
    exit;
fi

#cat train_text dev_text test_text > All_text

text_file=$1

lang_name='how2'

for no_of_tokens in 1000; do

    model_dir="models/${lang_name}_${no_of_tokens}"
    mkdir -p $model_dir
    model="${model_dir}/${lang_name}_${no_of_tokens}"

    name_suffix='.txt'

    #-----------
    cut -d " " -f1 $text_file>utt_id
    #-----------
    cut -d " " -f2- $text_file|sed 's/  */ /g' |  sort -u > utt_text
    #-----------


    #no_of_tokens=`cat utt_text|tr -s " " "\n"|sort|uniq|wc -l`
    #echo "$no_of_tokens"
    echo "$text_file"

    python utt_piece_training_nonorm.py utt_text $model $no_of_tokens $Special_string

done
