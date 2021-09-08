# /bin/sh

#----------------------------------------------------------
PPATH="/mnt/matylda4/kesiraju/code/ASR_Transformer"

export PYTHONUNBUFFERED=TRUE

only_scoring='False'
scoring_path='/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/scoring/'
stage=3

. path.sh


dataset="how2"

#----------------------------------------------------------
if [ $stage -le 3 ];
then

#TransLM_model="/mnt/matylda3/vydana/HOW2_EXP/IWSLT_LM/models/Trnas_eng_Lm_training_from_pretrained_librispeech_6L_1024_4096/model_epoch_8_sample_7999_61.70172444304824___71.48664149531612"
#TransLM_model="/mnt/matylda3/vydana/HOW2_EXP/IWSLT_LM/models/Trnas_eng_Lm_training_from_pretrained_librispeech_6L_1024_4096_pretrained_fromLibrispeech_finetuning/Trnas_eng_Lm_training_from_pretrained_librispeech_6L_1024_4096_pretrained_fromLibrispeech_finetuning_SWA_random_tag_12014_args_ealystpping_checkpoints_3"

TransLM_model="0"
RNNLM_model="0"

Word_model_path="/mnt/matylda3/vydana/kaldi/egs/IWSLT18/Tokenizers_en_utt_text.norm.lc.rm.tok/models/All_MT_files_for_SP_tokenization_en_utt_text.norm.lc.rm.tok_clean_format__len_pruned__en___noascii__40Msent_20000_bpe.model"
Char_model_path=$Word_model_path


if [[ $dataset == "Librispeech" ]];
then

    source /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/SCORING_HELPER_Librispeech.sh
    text_file="/mnt/matylda3/vydana/kaldi/egs/Mozilla_common_voice/Librispech_data_IWSLT/Librispeech_All_text_normalized_en_en_utt_text.norm.lc.rm.tok_clean_format"
    librispeech_path="/mnt/matylda3/vydana/espnet_latest/espnet_JAN2020/espnet/egs/librispeech/asr1/data_fbank249/scp_files/ALL_dev_test/"



#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_SWA_random_tag_12464_args_ealystpping_checkpoints_4"

    pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_SWA_random_tag_31222_args_ealystpping_checkpoints_8"

    max_jobs_to_decode=11126
    test_path="$librispeech_path"


elif [[ $dataset == "Mustc"  ]];
then

    source /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/SCORING_HELPER_Mustc.sh

    text_file='/mnt/matylda3/vydana/kaldi/egs/Textnorm/espnet_data_cleaning/All_Joint_files_for_SP_tokenization_en_utt_text.norm.lc.rm.tok_clean_format'
    #mustc_path='/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_JointSLT_systems/scp_files/Mustc_V2/dev/'
    mustc_path="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_JointSLT_systems/scp_files/Mustc_V2/dev_mustc/"




#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_SWA_random_tag_26587_args_ealystpping_checkpoints_4"

#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT_SWA_random_tag_6668_args_ealystpping_checkpoints_4"


#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT_SWA_random_tag_25377_args_ealystpping_checkpoints_8"


#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT_cleaned/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT_cleaned_SWA_random_tag_19086_args_ealystpping_checkpoints_8"





#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT_cleaned/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_finetune_mustc_ted_IWSLT_cleaned_SWA_random_tag_26612_args_ealystpping_checkpoints_4"



####SStraining###
pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_ASR/models/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_SSfinetuning_ASR_MT_0.5/Librispeech_Transformer_ASR_enc12_1024_4096_dec6_1024_4096_15K_accm1_lc_rm_16H_SSfinetuning_ASR_MT_0.5_SWA_random_tag_20448_args_ealystpping_checkpoints_6"




max_jobs_to_decode=2560
test_path="$mustc_path"

fi


AAAA="${pre_trained_weight%/*}"
model_file="${AAAA##*/}"
echo "$model_file"


model_dir="$PPATH/models/$model_file"
weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res

#SWA_random_tag="$RANDOM"ls
gpu=0
#######this should have at maximum number of files to decode if you want to decode all the file then this should be length of lines in scps
mem_req_decoding=50G
early_stopping_checkpoints=10
escp=$early_stopping_checkpoints


SWA_random_tag="$RANDOM"
#SWA_random_tag=12981


for test_fol in $test_path
do

echo "$max_jobs_to_decode"
echo "$test_fol"

#---------------------------
D_path=${test_fol%*/}
D_path=${D_path##*/}
echo "$test_fol"
echo "$D_path"


gamma=1.0
len_pen=0.5
#---------------------------------

beam=10
escp=10
apply_cmvn=1
#SWA_random_tag="$RANDOM"
Am_weight=1.0
len_bonus=0.7

for Am_weight in 1.0 #0.95 0.8 0.85 0.7 0.75
do
decoding_tag="_Librispeech_decoding_v1_beam_${beam}_${D_path}_gamm${gamma}_len_pen${len_pen}_${escp}_SWA_random_tag_${SWA_random_tag}_Am_weight_corr_Trans_LM_81849677_cp10_Am_weight_${Am_weight}_len_bonus_${len_bonus}__Translm_ppl_46_SWA_random_tag_12014"

log_path=${model_dir}/decoding_log_${decoding_tag}
echo "$log_path"

mkdir -pv "$log_path"
mkdir -pv "$log_path/scoring"
mkdir -pv "$model_dir/decoding_files/plots"

if [ $only_scoring != 'True' ];
then

for max_jobs in $max_jobs_to_decode
do
    ${PPATH}/queue.pl \
	--max-jobs-run $max_jobs_to_decode \
	-q all.q@@stable,all.q@@blade \
	--mem $mem_req_decoding \
	-l matylda4=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
	JOB=1:$max_jobs \
	-l 'h=!blade063' \
	$log_path/decoding_job.JOB.log \
	python ${PPATH}/ASR_TransV1/Transformer_Decoding.py \
	--gpu $gpu \
	--Decoding_job_no JOB \
	--beam $beam \
	--dev_path $test_fol \
	--SWA_random_tag $SWA_random_tag \
	--weight_text_file $weight_text_file\
        --Res_text_file $Res_text_file\
        --text_file $text_file \
        --pre_trained_weight $pre_trained_weight \
        --apply_cmvn $apply_cmvn \
        --model_dir $model_dir \
        --early_stopping_checkpoints $escp \
        --RNNLM_model $RNNLM_model \
        --Am_weight $Am_weight \
        --TransLM_model $TransLM_model \
        --len_pen $len_pen \
        --gamma $gamma \
        --Word_model_path $Word_model_path \
        --Char_model_path $Char_model_path \
        --len_bonus $len_bonus


done

echo "1"
ASR_Scoring $log_path $max_jobs_to_decode


else

ASR_Scoring $log_path $max_jobs_to_decode

echo "1"


fi
done
done

fi
