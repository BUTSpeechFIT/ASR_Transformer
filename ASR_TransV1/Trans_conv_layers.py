#! /usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 

#=============================================================================================================
class Conv_2D_Layers(nn.Module):
        def __init__(self,args):
                super(Conv_2D_Layers,self).__init__()
                self.input_size = int(args.input_size)

                ##get the output as the same size of encoder d_model
                self.hidden_size = int(args.encoder_dmodel)

                self.kernel_size = int(args.kernel_size)
                self.stride = args.stride
                self.in_channels = int(args.in_channels)
                self.out_channels = int(args.out_channels)
                self.conv_dropout  = args.conv_dropout 
                
                #dropout layer
                self.conv_dropout_layer = nn.Dropout(self.conv_dropout)
                
                ###two subsamling conv layers
                self.conv1=torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=1, dilation=1, groups=1, bias=True)
                self.conv2=torch.nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=1, dilation=1, groups=1, bias=True)
                
                linear_in_size=math.ceil(self.out_channels*(math.ceil(self.input_size/(self.stride*2))))
                ### makes the outputs as  (B * T * d_model)
                self.linear_out=nn.Linear(linear_in_size, self.hidden_size)
        
        def forward(self, input):

                CV1=F.relu(self.conv_dropout_layer(self.conv1(input.unsqueeze(1))))
                CV2=F.relu(self.conv_dropout_layer(self.conv2(CV1)))

                conv_output=CV2
                b, c, t, f = conv_output.size()
                conv_output=conv_output.transpose(1,2).contiguous().view(b,t,c*f)
                lin_conv_output=self.linear_out(conv_output)    
                return lin_conv_output

#---------------------------------------------------------------------------------------------------------------
#===============================================================================================================