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
from mamba_ssm.modules.mamba_simple import Mamba,Block
from model import ASC
from einops import rearrange, repeat, einsum


    

# set_seed(888)
class SIR_Block(nn.Module):
    def __init__(self,d_model,inchannels,J=6):
        super(SIR_Block, self).__init__() 
        k_size = 2 * J 
        self.conv1d1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k_size, stride=2, padding=(k_size - 2) // 2)
        self.conv1d2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k_size, stride=2, padding=(k_size - 2) // 2)
        self.ln1 = nn.LayerNorm(inchannels // 2)
        self.silu = nn.SiLU()
        self.ln2 = nn.LayerNorm(d_model)
        self.adpfactor = nn.Parameter(torch.tensor(1.0))  
    def forward(self, h): 
        delte_h = h.permute(0, 2, 1) 
        delte_h1 = self.conv1d1(delte_h) 
        delte_h2 = self.conv1d2(delte_h)
        delte_h1 = delte_h1.permute(0,2,1) #b,d,l//2 ->b,l//2,d
        delte_h2= delte_h2.permute(0,2,1) #b,d,l//2 ->b,l//2,d
        delte_h_w = torch.sigmoid(self.adpfactor * delte_h2) #b,l//2,d 
        x_out = delte_h1*delte_h_w
        x_out = self.ln2(x_out)
        x_out = self.silu(x_out) #b,d,l//2
        return x_out
#-------------Process_aware_Branch--------
class Process_aware_Branch(nn.Module):
    # input (b, L, 1)
    # need to squeeze the last dimension -> (b, L) then use conv2d -> (b, L) 
    # output (b, L) -> unsqueeze(-1) -> (b, L, 1)
    def __init__(self, channels, d_model):
        super(Process_aware_Branch, self).__init__()
        self.efc1 = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1, stride=1)  
        self.ly = nn.LayerNorm(channels)
        self.relu = nn.SiLU()
        self.efc2 = nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=1, stride=1)  
        self.sigmoid = nn.Sigmoid()
        self.adpfactor = nn.Parameter(torch.tensor(1.0))  

    def forward(self, x):  # x (b, l, d)
        # Statewise Enhance Module
        x = x.contiguous() 
        x_se = self.efc1(x.permute(0, 2, 1))  # bdl -> b1l
        x_se = torch.sigmoid(self.adpfactor * x_se)  
        x = x * (x_se.permute(0, 2, 1))  # (b, l, d)
        x = x.contiguous() 
        x_de = self.efc2(x)  # bld -> b1d
        x_de = torch.sigmoid(self.adpfactor * x_de) 
        x_de = x_de.contiguous() 
        x = (x.permute(0, 2, 1)) * (x_de.permute(0, 2, 1))  # bdl * b1d
        x = x.permute(0, 2, 1)  # bld
        return x

#-------------mamba-block-----------
class MambaBlock(nn.Module):
    def __init__(self,L,d_model,d_State,d_conv=4):
        super(MambaBlock,self).__init__()
        self.Process_aware_Branch = Process_aware_Branch(L,d_model)
        self.ln = nn.LayerNorm(d_model)
        self.Silu = nn.SiLU()
        self.mamba = Mamba(
            d_model=d_model , #input ssm dimension
            d_state=d_State, # state dimension
            d_conv=d_conv, #1D-convolution filter size
        )
        # self.pool = nn.AvgPool1d(L,1)
    def forward(self,x):
        residual = self.Process_aware_Branch(x) #(b,L,d) 
        residual = self.ln(residual) #(b,L,1)->(b,L,1)
        residual = self.Silu(residual)  #(b,L,1)->(b,L,1)
        x = self.mamba(x) #(b,L,d)->(b,L,d)
        x = self.ln(x) #(b,L,d)->(b,L,d)
        x = self.Silu(x) + residual #bld
        return x
    

    
class pool(nn.Module):
    #input x(b,l,d) ->(b,l//2,d)
    def __init__(self,kernel,stride):
        super(pool,self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size=kernel,stride=stride)
    def forward(self,x):#
        x = x.permute(0,2,1) 
        x = self.avgpool(x)   #bdl ->b,d,l//2
        x = x.permute(0,2,1) #b,l//2,d
        return x


class MioA(nn.Module):
    #input x(b,l,d) ->(b,l//2,d)
    def __init__(self,d_model,d_State,d_conv,L,P,unbed):
        super(MioA,self).__init__()
        self.rnn_forward = nn.RNN(d_model, d_model, 1, batch_first=True)
        self.rnn_backword = nn.RNN(d_model, d_model, 1, batch_first=True)
        self.rnn_mid = nn.RNN(d_model, d_model, 1, batch_first=True)
        self.adaptive = nn.Linear(d_model,1,bias=False)
        self.silu = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)
        self.P = P
        self.unbed = unbed
        self.MioA_auto = MioA_auto(d_model=d_model,d_state=d_State,d_conv=d_conv,P=P)
    def forward(self,x):#bl//16d
        x6 = x
        y = self.MioA_auto(x)
        if self.unbed == 'linear':
            y = self.adaptive(y)
        elif self.unbed == 'max':
            y = y.max(dim=-1).values
        elif self.unbed == 'mean':
            y = y.mean(dim=-1)
        # out = self.adaptive(0.5 * out_forward+ 0.5 * out_backward)
        #     outputs.append(out)
        # outputs =torch.cat(outputs, dim=1)
        return y.squeeze(),x6
        
    
    

class ProMU(nn.Module):
    """
    input x(b,l,1),  
    L is the number of bands
    P is the number of endmembers
    """
    def __init__(self,L,P,drop,d_model,d_State,d_conv=4,J=6,unemb='max'):
        super(ProMU,self).__init__()
        # self.embedding = nn.Sequential(
        #     ,
        #     nn.LayerNorm(L),
        #     nn.SiLU(),
        #     nn.Dropout(drop),
            # )
        self.conv1d = nn.Conv1d(in_channels=1,out_channels=d_model,kernel_size=1,stride=1)  #(bl1)->bld
        self.ln = nn.LayerNorm(d_model) 
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.mambablock0 = MambaBlock(L,d_model,d_State,d_conv) #bld->bld
        # self.drop = nn.Dropout(drop)
        self.SIR0 = SIR_Block(d_model,L,J)#bld->bl//2d
        self.ap = pool(kernel=2,stride=2)#bld=>bl//2d
        self.mambablock1 =MambaBlock(L // 2,d_model,d_State,d_conv)
        self.SIR1 = SIR_Block(d_model,L//2,J)
        self.mambablock2 = MambaBlock(L // 4,d_model,d_State,d_conv)#(b,l//4) ->(b,l//4)
        self.SIR2 = SIR_Block(d_model,L//4,J)
        self.mambablock3 = MambaBlock(L // 8,d_model,d_State,d_conv)
        self.SIR3 = SIR_Block(d_model,L//8,J)
        self.mambablock4 = MambaBlock(L // 16,d_model,d_State,d_conv) #(b,l//16,d)
        self.SIR4 = SIR_Block(d_model,L//16,J)
        self.MioA_auto = nn.Sequential(MambaBlock(L // 32,d_model,d_State,d_conv),
                                 MioA(d_model,d_State,d_conv,L//32,P,unemb))
        self.ly = nn.LayerNorm(P)
        self.ReLU = nn.ReLU() 
        self.ASC = ASC()
        self.decoder = nn.Sequential(
            nn.Linear(P,L,bias=False),
            )

    def forward(self,x):
        x = torch.unsqueeze(x,-1) #(b,l) ->(b,l,1)
        x = x.permute(0,2,1)
        x = self.conv1d(x) #(b,1,l) ->#(b,16,l)
        x0 = x.permute(0,2,1)  #(b,1,l) ->#(b,l,16)
        x = self.drop(x0)
        x = self.ln(x)
        x = self.silu(x)
        x = self.mambablock0(x) #(b,l,16) ->#(b,l,16)
      
        x1 = self.ap(x)+self.SIR0(x0) #(b,l//2,16) ->(b,l//2,16)
        x = self.mambablock1(x1) #(b,l//2,16) ->(b,l//2,16)
        x2 = self.ap(x)+self.SIR1(x1) #(b,l//2,16) ->(b,l//4,16)
        x = self.mambablock2(x2)#->(b,l//4,16->(b,l//4,16)
        x3 = self.ap(x)+self.SIR2(x2) #->(b,l//4,16) ->->(b,l//8,16)
        x = self.mambablock3(x3) #->((b,l//8,16) ->((b,l//8,16))
        x4 = self.ap(x)+self.SIR3(x3) #->((b,l//8,16) ->((b,l//16,16))
        x = self.mambablock4(x4) #->((b,l//16,16) ->((b,l//16,16))
        x5 = self.ap(x)+self.SIR4(x4)#->((b,l//16,16) ->((b,l//32,16))
        x,x6 = self.MioA_auto(x5) #->((b,l//16,16) ->((b,l//16,16))
        # x = self.adpstate(x6) #->((b,l//16,16) ->((b,P)
        x = self.ly(x)
        x = self.ReLU(x)
        abu_est = self.ASC(x)
        re_result = self.decoder(abu_est)
        return abu_est, re_result,x6#
    

class MioA_auto(nn.Module):
    def __init__(self, d_model,d_state,d_conv,P):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.expand = 1
        self.P = P
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(self.d_model / 16)
        self.bias = False
        self.conv_bias = True
        self.d_inner  = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 3, bias=False)
        self.xb_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        # x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        # (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # x = rearrange(x, 'b l d_in -> b d_in l')
        # x = self.conv1d(x)[:, :, :l]
        # x = rearrange(x, 'b d_in l -> b l d_in')
        
        # x = F.silu(x)

        y = self.ssm(x)
        
        # y = y * F.silu(res)
        
        y = self.out_proj(y)

        return y

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        Df = self.D.float()
        Db = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        delta_b = self.xb_proj(torch.flip(x, dims=[1]))
        (delta, Bf,Bb, C) = x_dbl.split(split_size=[self.dt_rank, n, n,n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        delta_b =  F.softplus(self.dt_proj(delta_b))
        y = self.selective_scan(x, delta,delta_b, A, Bf, Bb, C, Df, Db)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y

    
    def selective_scan(self, u, delta,delta_b, A, Bf, Bb, C, Df, Db):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaBf_u = einsum(delta, Bf, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        deltaBb_u = einsum(delta_b, Bb, torch.flip(u, dims=[1]), 'b l d_in, b l n, b l d_in -> b l d_in n')
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(self.P):
            x = deltaA[:, i] * x + deltaBf_u[:, i] + deltaBb_u[:,i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u[:,0:self.P,:] * Db + torch.flip(u, dims=[1])[:,0:self.P,:] * Df

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        