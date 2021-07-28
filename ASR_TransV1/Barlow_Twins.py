#!/usr/bin/python

#######################################################################
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable
#######################################################################


# def barlow_loss(B,C,lamda):
#       Batch_size,_, Dim = B.size()
        
#       batch_correlation=torch.matmul(C,B.transpose(1,2))/Batch_size

#       I=torch.eye(Dim)
#       IR=torch.cat([I.unsqueeze(0)]*Batch_size,dim=0)

#       batch_correlation_loss=(batch_correlation-IR).pow(2)
#       #breakpoint()
        

#       off_diag = lamda*(batch_correlation_loss*(1-IR)).sum()
#       diag = (IR*batch_correlation_loss).sum()
#       similarity = off_diag +  diag
        
#       print('off_diag',off_diag,'diag',diag)
#       #print(diag<off_diag/20)
#       return similarity

#######################################################################
#==========================================================================
def cal_performance(pred, gold,IGNORE_ID,normalize_length=False,smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:  pred: N x T x C, score before softmax ;;;; gold: N x T """
    pred=pred.unsqueeze(1)
    gold=gold.unsqueeze(1)
    #breakpoint()

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)

    loss = cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing)
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)

    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    n_correct=n_correct/float(non_pad_mask.sum())

    n_correct=1.0-n_correct
    return loss, n_correct
#===============================================
#===============================================
def cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing):
    """Calculate cross entropy loss, apply label smoothing if needed.  """
    normalize_length=True
    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        #breakpoint()
        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class

        #breakpoint()


        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)   

        loss = loss.masked_select(non_pad_mask).sum() / n_word
        print('CE_loss',loss)
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')
    return loss

#######################################################################
def barlow_loss_VICReg(z_a,z_b,lamda):
        """
        x_a, x_b = augment(x)
        #compute representations
        z_a = f(x_a) # N x D
        z_b = f(x_b) # N x D
        # invariance loss
        sim_loss = mse_loss(z_a, z_b)
        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(relu(1 - std_z_a))
        std_loss = std_loss + torch.mean(relu(1 - std_z_b))
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D
        cov_loss = cov_loss + off_diagonal(cov_z_b).pow_(2).sum() / D
        # loss
        loss = lambda * sim_loss + mu * std_loss + nu * cov_loss
        # optimization step
        loss.backward()
        optimizer.step()
        """
        lamda,mu,nu=0.1,0.1,0.1
        N,D = z_a.size()
        mse_loss = torch.nn.MSELoss(reduction='none')   
        sim_loss = mse_loss(z_a, z_b)

        ###mean keeps loss independent of proj_dim
        sim_loss = sim_loss.mean(dim=1)

        std_z_a = torch.sqrt(torch.var(z_a,dim=0) + 1e-04)
        std_z_b = torch.sqrt(torch.var(z_b,dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        
        z_a = z_a -torch.mean(z_a, dim=0, keepdim=True)        
        z_b = z_b -torch.mean(z_b, dim=0, keepdim=True)

        cov_z_a = torch.mm(z_a.transpose(0,1),z_a)/ (N - 1)
        cov_z_b = torch.mm(z_b.transpose(0,1),z_b)/ (N - 1)
        
        IR=torch.cat([torch.eye(D).unsqueeze(0)]*N,dim=0)
        
        offdiag_a = ((cov_z_a*(1-IR))**2).sum(dim=1) / D
        offdiag_b = ((cov_z_b*(1-IR))**2).sum(dim=1) / D

        cov_loss = offdiag_a + offdiag_b
        cov_loss = cov_loss.sum(dim=1)

        loss = lamda * sim_loss + mu * std_loss + nu * cov_loss
        loss = loss.sum()
        #print('loss',loss,',sim_loss,',sim_loss,'std_loss', std_loss,'cov_loss', cov_loss)
        return loss
#######################################################################
class Barlow_CE_Loss(nn.Module):
        def __init__(self,input_dim,proj_dim):
                super(Barlow_CE_Loss,self).__init__()
                self.Linear_proj=nn.Linear(input_dim,proj_dim)
                self.barlow_loss=barlow_loss_VICReg
                self.cal_performance=cal_performance

        def forward(self,B,C,D):
                z_a = self.Linear_proj(B)
                z_b = self.Linear_proj(C)
                Bar_los = self.barlow_loss(z_a,z_b,1)
                self.cal_performance(B,D,IGNORE_ID=1024,normalize_length=False,smoothing=0.1)
                print(Bar_los)
#######################################################################

loss=Barlow_CE_Loss(1024,200)
for i in range(1,100):
        B=torch.randn(2,1024)
        C=torch.randn(2,1024)
        D=torch.randint(low=0, high=1024,size=(2,1))
        # C=-1*B
        #barlow_loss(B,C,1)
        #barlow_loss_VICReg(B,C,1)
        loss(B,C,D)
