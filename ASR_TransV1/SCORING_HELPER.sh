#!/usr/bin/bash

if [ $# -ne 1 ]; then
    echo "$0 <log_path>"
    exit
fi

PRE=/mnt/matylda4/kesiraju/code/ASR_Transformer

scoring_path=${PRE}/ASR_TransV1/scoring

log_path=$1

cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $3}'| tr -s " "> $log_path/scoring/hyp_val_file
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $4}'| tr -s " "> $log_path/scoring/ref_val_file


cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $3, "(" $2 ")"}'| tr -s " "> $log_path/scoring/hyp_val_file_sc
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $4, "(" $2 ")"}'| tr -s " "> $log_path/scoring/ref_val_file_sc

bash $scoring_path/comput_wer_sclite.sh "$log_path/scoring/hyp_val_file" "$log_path/scoring/ref_val_file" "$log_path/scoring" "$log_path/scoring/hyp_val_file_sc" "$log_path/scoring/ref_val_file_sc"

cat $log_path/scoring/wer_val
