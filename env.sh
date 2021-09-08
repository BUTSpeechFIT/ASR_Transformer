
host=`hostname -d`

if [ "${host}" == "fit.vutbr.cz" ]; then

    source /mnt/matylda4/kesiraju/envs/pt18/bin/activate
    export DS_HOME="/mnt/matylda4/kesiraju/datasets/"
    export HF_HOME="/mnt/matylda4/kesiraju/hugging-face/"

else

    source ~/envs/pt18/bin/activate

fi
