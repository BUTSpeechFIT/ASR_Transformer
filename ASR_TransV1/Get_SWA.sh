#! usr/bin/bash




model_dir="/mnt/matylda3/vydana/HOW2_EXP/Librispeech_V2/models/LIB_Trans_100M_12_6_512_8_2048_dr0.1_noclipgrad_accmgrad4"
SWA_random_tag="$RANDOM"
est_cpts=4

python /mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer/ASR_TransV1/Get_SWA_weights.py $model_dir $SWA_random_tag $est_cpts
