#! /usr/bin/python

import sys
import os
import subprocess
import torch

def Set_gpu():
        cuda_command = 'nvidia-smi --query-gpu=memory.free,memory.total --format=csv | tail -n+2 | ' \
               'awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''

        #os.environ["CUDA_VISIBLE_DEVICES"] = str(subprocess.check_output(cuda_command, shell=True)).rsplit('\n')[0]
        oooo=subprocess.check_output(cuda_command, shell=True)
        print(oooo)

        dev_id=str(oooo).lstrip('b').strip("'").split('\\n')[0]
        os.environ["CUDA_VISIBLE_DEVICES"]=dev_id

        print(os.environ["CUDA_VISIBLE_DEVICES"])
        gpu_no=os.environ["CUDA_VISIBLE_DEVICES"]
        zzz=torch.zeros(10,10)
        zzz=zzz.cuda()


