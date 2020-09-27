#!/usr/bin/bash



#cat train_text dev_text test_text > All_text



text_file=All_text
for no_of_tokens in 100 
do 

model="models/Timit_PHSEQ_$no_of_tokens/Timit_PHSEQ__"$no_of_tokens"_"

name_suffix='.txt'

#-----------
cut -d " " -f1 $text_file>utt_id
#-----------
cut -d " " -f2- $text_file|sed 's/  */ /g'>utt_text
#-----------


no_of_tokens=`cat utt_text|tr -s " " "\n"|sort|uniq|wc -l`
echo "$no_of_tokens"
echo "$text_file"
mkdir -pv $model
python utt_piece_training_nonorm.py utt_text $model $no_of_tokens $Special_string
done




