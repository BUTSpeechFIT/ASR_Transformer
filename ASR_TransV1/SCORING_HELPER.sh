#!/usr/bin/bash 


log_path=$1

cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $3}'| tr -s " "> $log_path/scoring/hyp_val_file
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $4}'| tr -s " "> $log_path/scoring/ref_val_file


cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $3, "(" $2 ")"}'| tr -s " "> $log_path/scoring/hyp_val_file_sc
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $4, "(" $2 ")"}'| tr -s " "> $log_path/scoring/ref_val_file_sc

bash $scoring_path/comput_wer_sclite.sh "$log_path/scoring/hyp_val_file" "$log_path/scoring/ref_val_file" "$log_path/scoring" "$log_path/scoring/hyp_val_file_sc" "$log_path/scoring/ref_val_file_sc"

cat $log_path/scoring/wer_val
