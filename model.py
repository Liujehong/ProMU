import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from Toolbox.unmix_utils import *
import scipy.io as sio
import numpy as np
import os
import math
import time
import random
from Toolbox.result_displayforpy import aRMSE,rRMSE,SAM
from torch.autograd import Variable, Function


#GaussianDropout
class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        # Constructor
        super(GaussianDropout, self).__init__()
        
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x






#
class CycUnet(nn.Module):
    """
    L:num_bands
    P:num_abu(num_endmembers)
    drop:dropout rate
    """
    def __init__(self,L,P,drop):
        super(CycUnet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Conv2d(128, 64,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, P, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(P,momentum=0.9),
            nn.Softmax(dim=1)
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
    def forward(self,x):
        abu_est1 = self.encoder(x)
        # abu_est1 = self.encoder(x)
        re_result1 = self.decoder1(abu_est1)
        abu_est2 = self.encoder(re_result1)
        re_result2 = self.decoder2(abu_est2)
        return abu_est1, re_result1, abu_est2, re_result2
        # return abu_est1, re_result1



class DConvAEN(nn.Module):
    """
    L:num_bands
    P:num_abu(num_endmembers)
    drop:dropout rate
    """
    def __init__(self,L,P,drop):
        super(DConvAEN, self).__init__()
        self.encoder = nn.Sequential( nn.Conv2d(L, 128,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Conv2d(128, 64,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, P, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(P,momentum=0.9),
            # nn.ReLU()
            nn.Softmax(dim=1)
        )
        self.decoder = nn.Sequential(nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),)
        
    def forward(self,x):
        abu_est = self.encoder(x)
        re_result = self.decoder(abu_est)
        return abu_est, re_result


class DAEN(nn.Module):
    """
    L:num_bands
    P:num_abu(num_endmembers)
    drop:dropout rate
    """
    def __init__(self,L,P,drop):
        super(DAEN, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(L,9*P),
                                     nn.Dropout(drop),
                                     nn.Linear(9*P,6*P),
                                     nn.Linear(6*P,3*P),
                                     nn.Linear(3*P,P),
                                     nn.BatchNorm1d(P),
                                     nn.LeakyReLU(),
                                     ASC(),
                                     )
        self.decoder = nn.Sequential(nn.Linear(P,L,bias=False),
                                     )
        
    def forward(self,x):
        abu_est = self.encoder(x)
        re_result = self.decoder(abu_est)
        return abu_est, re_result
    




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function

class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        # Constructor
        super(GaussianDropout, self).__init__()
        
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x
            

class ASC(nn.Module):
  def __init__(self):
    super(ASC, self).__init__()
  
  def forward(self, input):
    """Abundances Sum-to-One Constraint"""
    constrained = input/((torch.sum(input, dim=1)).unsqueeze(1).expand(input.shape))
    return constrained
  





import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SAD(nn.Module):
  def __init__(self, num_bands: int=156):
    super(SAD, self).__init__()
    self.num_bands = num_bands

  def forward(self, input, target):
    """Spectral Angle Distance Objective
    Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'
    
    Params:
        input -> Output of the autoencoder corresponding to subsampled input
                tensor shape: (batch_size, num_bands)
        target -> Subsampled input Hyperspectral image (batch_size, num_bands)
        
    Returns:
        angle: SAD between input and target
    """
    try:
      input_norm = torch.sqrt(torch.bmm(input.view(-1, 1, self.num_bands), input.view(-1, self.num_bands, 1)))
      target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1)))
      
      summation = torch.bmm(input.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
      angle = torch.acos(summation/((input_norm * target_norm)+1e-9))
      
    
    except ValueError:
      return 0.0
    
    return angle

class SID(nn.Module):
  def __init__(self, epsilon: float=1e5):
    super(SID, self).__init__()
    self.eps = epsilon

  def forward(self, input, target):
    """Spectral Information Divergence Objective
    Note: Implementation seems unstable (epsilon required is too high)
    Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'
    
    Params:
        input -> Output of the autoencoder corresponding to subsampled input
                tensor shape: (batch_size, num_bands)
        target -> Subsampled input Hyperspectral image (batch_size, num_bands)
        
    Returns:
        sid: SID between input and target
    """
    normalize_inp = (input/torch.sum(input, dim=0)) + self.eps
    normalize_tar = (target/torch.sum(target, dim=0)) + self.eps
    sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) + normalize_tar * torch.log(normalize_tar / normalize_inp))
    
    return sid