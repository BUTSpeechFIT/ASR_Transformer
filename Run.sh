#! /bin/sh
#
#$ -N neur3_exp
#$ -q long.q@supergpu*,long.q@facegpu*,long.q@pc*,long.q@dellgpu*
#$ -l gpu=1,gpu_ram=7G,ram_free=7G,matylda4=0.5

#$ -o /mnt/matylda4/xsarva00/NEUREM3/HOW2_EXP/ASR_Transformer/log/Transformer_T4.log
#$ -e /mnt/matylda4/xsarva00/NEUREM3/HOW2_EXP/ASR_Transformer/log/Transformer_T4.err

ulimit -t 46800

if [ $# -ne 1 ]; then
    echo "$0 <stage>"
   # exit
fi

PPATH="/mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer"

. ${PPATH}/env.sh
# cd "$PPATH"
export PYTHONUNBUFFERED=TRUE

only_scoring='False'
#scoring_path='/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/scoring/'

stage=2


gpu=1
max_batch_len=10000
max_batch_label_len=5000

tr_disp=50
vl_disp=10
validate_interval=500
max_val_examples=400

compute_ctc=0
ctc_weight=0.3

learning_rate=0.0001
early_stopping=0
clip_grad_norm=5


# input featsize (depending of MFCC 13, or FBANK 83 + pitch)
# --- IMPORTANT
input_size=83


hidden_size=256
kernel_size=3
stride=2
in_channels=1
out_channels=64
conv_dropout=0.3
isresidual=0
label_smoothing=0.1

max_feat_len=3000
max_label_len=400

####encoder parameters
encoder_layers=4
encoder_dmodel=256
encoder_heads=4
encoder_dinner=1024
encoder_dropout=0.1
encoder_ff_dropout=0.3

####decoder parameters
dec_embd_vec_size=256
decoder_dmodel=256
decoder_heads=4
decoder_dinner=1024
decoder_dropout=0.1
decoder_ff_dropout=0.3
decoder_layers=4
tie_dec_emb_weights=0

warmup_steps=5000

teacher_force=0.6
min_F_bands=5
max_F_bands=30
time_drop_max=2
time_window_max=1

weight_noise_flag=0
reduce_learning_rate_flag=0
spec_aug_flag=0

#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/models/TIMIT_fullnewsetup_2_4dr0.3_LAS_loc/model_epoch_13_sample_6501_40.030520648190546___1500.3319549560547__0.23846153846153847"
pre_trained_weight="0"

plot_fig_validation=0
plot_fig_training=0
start_decoding=0

#============================================================================================
#---------------------------
Word_model_path=${PPATH}/Tokenizers/models/how2_1000/how2_1000_char.model
Char_model_path=${PPATH}/Tokenizers/models/how2_1000/how2_1000_char.model

text_file='/mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/Tokenizers/how2_train_val_text'

# path to feats.scp is in these folders
train_path='/mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_feats_scp/train/'
dev_path='/mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_feats_scp/val/'
test_path='/mnt/matylda4/xsarva00/NEUREM3/ASR_Transformer/how2_feats_scp/dev5/'

#==========================================================================================


data_dir="$PPATH/how2_data_prep/"
mkdir -pv $data_dir

model_file="Transformer_T4_how2"
mkdir -p ${PPATH}/models/how2
model_dir="$PPATH/models/how2/$model_file"

mkdir -p ${PPATH}/weight_files/how2/
weight_text_file="${PPATH}/weight_files/how2/$model_file"
Res_text_file="${PPATH}/weight_files/how2/$model_file"_Res

mkdir -pv $model_dir
mkdir -p ${PPATH}/weight_files/

output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log

if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi

echo "model dir       : $model_dir"
echo "weight text file: $weight_text_file"
echo "res text file   : $Res_text_file"
echo "data dir        : ${data_dir}"
echo "text file       : ${text_file}"
#====================================================================================

if [ $stage -le 1 ]; then
# #---------------------------------------------------------------------------------------------
    ##### making the data preperation for the experiment
    echo "---------------------------"
    echo "     Creating scp files    "
    echo "---------------------------"
    stdbuf -o0  python ${PPATH}/ASR_TransV1/Make_training_scps.py \
		   --data_dir $data_dir \
 		   --text_file $text_file \
 		   --train_path $train_path \
 		   --dev_path $dev_path \
 		   --Word_model_path $Word_model_path \
 		   --Char_model_path $Char_model_path

    # Split the train/feats.scp into 100 lines per file and save them in the following dir
    # split -l 100 how2_feats_scp/train/feats.scp how2_data_prep/train/
    # same is done for val/feats.scp
##---------------------------------------------------------------------------------------------
fi


if [ $stage -le 2 ]; then

    echo "---------------------------"
    echo "         Training          "
    echo "---------------------------"


    # #---------------------------------------------------------------------------------------------
    stdbuf -o0  python ${PPATH}/ASR_TransV1/Transformer_training.py \
		   --data_dir $data_dir \
 		   --gpu $gpu \
 		   --text_file $text_file \
 		   --train_path $train_path \
 		   --dev_path $dev_path \
 		   --Word_model_path $Word_model_path \
 		   --Char_model_path $Char_model_path \
 		   --max_batch_label_len $max_batch_label_len \
 		   --tr_disp $tr_disp \
 		   --validate_interval $validate_interval \
 		   --weight_text_file $weight_text_file \
 		   --Res_text_file $Res_text_file \
 		   --model_dir $model_dir \
 		   --max_val_examples $max_val_examples \
 		   --compute_ctc $compute_ctc \
 		   --ctc_weight $ctc_weight \
 		   --spec_aug_flag $spec_aug_flag \
 		   --in_channels $in_channels \
 		   --out_channels $out_channels \
 		   --learning_rate $learning_rate \
 		   --early_stopping $early_stopping \
 		   --vl_disp $vl_disp \
 		   --clip_grad_norm $clip_grad_norm \
 		   --label_smoothing $label_smoothing \
		   --input_size $input_size \
 		   --min_F_bands $min_F_bands \
 		   --max_F_bands $max_F_bands \
 		   --time_drop_max $time_drop_max \
 		   --time_window_max $time_window_max \
 		   --pre_trained_weight $pre_trained_weight \
 		   --plot_fig_validation $plot_fig_validation \
 		   --plot_fig_training $plot_fig_training \
 		   --encoder_layers $encoder_layers \
		   --encoder_dmodel $encoder_dmodel \
		   --encoder_heads $encoder_heads \
		   --encoder_dinner $encoder_dinner \
		   --encoder_dropout $encoder_dropout \
		   --encoder_ff_dropout $encoder_ff_dropout \
		   --dec_embd_vec_size $dec_embd_vec_size \
		   --decoder_dmodel $decoder_dmodel \
		   --decoder_heads $decoder_heads \
		   --decoder_dinner $decoder_dinner \
		   --decoder_dropout $decoder_dropout \
		   --decoder_ff_dropout $decoder_ff_dropout \
		   --decoder_layers $decoder_layers \
		   --tie_dec_emb_weights $tie_dec_emb_weights \
		   --warmup_steps $warmup_steps \
           --max_feat_len $max_feat_len \
           --max_label_len $max_label_len

    #---------------------------------------------------------------------------------------------
fi
exit;
if [ $stage -le 3 ]; then
    bash ASR_Trans_V1/Get_SWA.sh
    ASR_TransV1/Get_SWA.sh
fi
set -e
set -x
if [ $stage -le 4 ]; then

    RNNLM_model="0"
    TransLM_model="0"
    gamma=1.0
    len_pen=0.5

    escp=10
    apply_cmvn=1
    Am_weight=1.0
    len_bonus=0.7

    #===========================================================
    SWA_random_tag="$RANDOM"
    gpu=0
    #######this should have at maximum number of files to decode
    #if you want to decode all the file then this should be length of lines in scps

    max_jobs_to_decode=10
    mem_req_decoding=10G
    #============================================================
    for test_fol in $dev_path
    do
        D_path=${test_fol%*/}
        D_path=${D_path##*/}
        echo "$test_fol"
        echo "$D_path"
        #============================================================
        for beam in 10
        do
            decoding_tag="_decoding_v1_beam_$beam""_$D_path"
            log_path="$model_dir"/decoding_log_$decoding_tag
            echo "$log_path"

            mkdir -pv "$log_path"
            mkdir -pv "$log_path/scoring"
            pre_trained_weight="/mnt/matylda4/kesiraju/code/ASR_Transformer/models/how2/Transformer_T4_how2/Transformer_T4_how2_SWA_random_tag_28922_args_ealystpping_checkpoints_8"
            #============================================================
            if [ $only_scoring != 'True' ]; then
                #============================================================
                    # ${PPATH}/queue.pl \
	                #         --max-jobs-run $max_jobs_to_decode \
	                #         -q short.q@@stable,short.q@@blade \
	                #         --mem $mem_req_decoding \
	                #         -l matylda4=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
	                #         JOB=1:$max_jobs_to_decode \
	                #         -l 'h=!blade063' \
	                #         $log_path/decoding_job.JOB.log \
	                        /mnt/matylda4/kesiraju/envs/pt18/bin/python3 ${PPATH}/ASR_TransV1/Transformer_Decoding.py \
	                        --gpu $gpu \
	                        --Decoding_job_no 1 \
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


                # . /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
                # bash ${PPATH}/ASR_TransV1/SCORING_HELPER.sh $log_path
                #============================================================
            # else

                # . /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
                # bash ${PPATH}/ASR_TransV1/SCORING_HELPER.sh $log_path

            fi

        done
    done

fi

###---------------------------------------------------------------------------------------------
