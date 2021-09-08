# "queue.pl" uses qsub. The options to it are options to qsub.

### DEFAULT CONFIGURATION:
# Run with qsub (the reources are typically site-dependent),
export train_cmd="queue.pl -l ram_free=1.5G,mem_free=1.5G"
export plain_cmd="$train_cmd"
export decode_cmd="queue.pl -l ram_free=2.5G,mem_free=2.5G"
export cuda_cmd="queue.pl -l gpu=1"


# Run locally,
#export plain_cmd=run.pl
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl

### Other tools

### SITE-SPECIFIC CONFIGURATION (OVERRIDES DEFAULT):
case "$(hostname -d)" in
  "fit.vutbr.cz") # BUT cluster,
    declare -A user2matylda=([iveselyk]=matylda5 [karafiat]=matylda3 [ihannema]=matylda5 [vydana]=matylda3 [kesiraju]=matylda4)
    matylda=${user2matylda[$USER]}
    queue="all.q@@blade"
    gpu_queue="long.q@@dellgpu*,long.q@@supergpu[1235678],long.q@pcspeech-gpu,long.q@@facegpu*"
    export plain_cmd="run.pl" # Runs locally (initial GMM training),
    export train_cmd="queue.pl -q $queue -l ram_free=1.5G,mem_free=1.5G,${matylda}=5"
    export decode_cmd="queue.pl -q $queue -l ram_free=2.5G,mem_free=2.5G,${matylda}=0.1"
    export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1 -l ram_free=5G,mem_free=5G"
  ;;
esac
