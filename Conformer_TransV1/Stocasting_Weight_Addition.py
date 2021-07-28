import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#IGNORE_ID=5000
from torch.autograd import Variable
from copy import deepcopy

import sys 
import os
import subprocess
from os.path import join, isdir



#from EncoderLayer import EncoderLayer
#from PositionalEncoding import PositionalEncoding
#from PositionwiseFeedForward import PositionwiseFeedForward


#------------------------------------------------------------------------------------------------------------
#=============================================================================================================
def Stocasting_Weight_Addition(model,model_names):

        noof_chkpnts = len(model_names)
        present_model = model_names[0]
        #print("Adding:",present_model)
        P_model_path=join(present_model)        
        model.load_state_dict(torch.load(P_model_path, map_location=lambda storage, loc: storage),strict=True)
        model_keys=model.state_dict().keys()
        #-------------------------
        for i in model_names[1:]:
            print("Adding:",i)
            P_model_path=join(i)

            new_params = torch.load(P_model_path, map_location=lambda storage, loc: storage)
            #----------------------------------------------------------------------------------------------
            ###This is for computing the sum of parameters over checkpoints
            for swa_param_key in model_keys:
                model.state_dict()[swa_param_key] += new_params[swa_param_key]

        ####This is for computing sum/number, it should iterate after summing all the models
        for swa_param_key in model_keys:
            model.state_dict()[swa_param_key] /= (noof_chkpnts)


        return model
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------



