#! /bin/bash
#
#$ -N n3_decoding
#$ -q long.q@@blade
#$ -l ram_free=7G,mem_free=7G,matylda4=0.5

#$ -o /mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_dev5_decoding.log 
#$ -e /mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_dev5_decoding.err 

ulimit -t 32400 

#if [ $# -ne 1 ]; then
    #echo "+ usage: $0 <job_num or utt number>"
    #echo $#
    #exit;
#fi

PRE=/mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer
FILE=2
#job_num=$1


scp_file='val'
if [ $FILE -ne 1 ]; 
then
   scp_file='dev5'
fi

echo "Data used for decoding: /mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_feats_scp/$scp_file/feats.scp"

num_jobs=`cat /mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_feats_scp/$scp_file/feats.scp | wc -l`

for ((job_num=0;job_num<num_jobs;job_num++));
do
echo "Decoding utt num: ${job_num}"
/mnt/matylda4/xsarva00/anaconda3/envs/neur3/bin/python3.9 \
    ${PRE}/ASR_TransV1/Transformer_Decoding.py \
    --gpu 0 \
    --Decoding_job_no ${job_num} \
    --beam 10 \
    --dev_path ${PRE}/how2_feats_scp/${scp_file}/ \
    --SWA_random_tag 21766 \
    --weight_text_file ${PRE}/weight_files/how2/Transformer_T4_how2 \
    --Res_text_file ${PRE}/weight_files/how2/Transformer_T4_how2_Res \
    --text_file ${PRE}/Tokenizers/how2_train_val_text \
    --pre_trained_weight ${PRE}/models/how2/Transformer_T4_how2/Transformer_T4_how2_SWA_random_tag_20886_args_ealystpping_checkpoints_8 \
    --apply_cmvn 1 \
    --model_dir ${PRE}/models/how2/Transformer_T4_how2 \
    --early_stopping_checkpoints 10 \
    --RNNLM_model 0 \
    --Am_weight 1.0 \
    --TransLM_model 0 \
    --len_pen 0.5 \
    --gamma 1.0 \
    --Word_model_path ${PRE}/Tokenizers/models/how2_1000/how2_1000_char.model \
    --Char_model_path ${PRE}/Tokenizers/models/how2_1000/how2_1000_char.model \
    --len_bonus 0.7
done
