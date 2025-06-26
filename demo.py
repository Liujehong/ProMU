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
from Toolbox.result_displayforpy import aRMSE,rRMSE,SAM,eSAM
from model import CycUnet,DConvAEN,DAEN,SAD,SID
from thop import profile
# from mambaunmix import CMUNet
# from mambaunmix_nopool import CMUNet_nopool
from mambaunmix_withse import *
from sklearn.preprocessing import minmax_scale

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#ss: 0.1244 | sumtoone loss: 0.0000 | re loss: 0.1244 aRMSE: 0.08742 | eSAM:0.07358 | rSAM:0.11581

# Load Data
dataset = 'samson' #samson |jasper | urban | apex | dc
model_name = 'ProMU'  #DAEN | DConvAEN |CycUnet |CMUNet |CMUNet_se
loss = 'MSE_SAD' #MSE |SAD | SID | MSE_SAD | top_K_SAD_MSE

 
#P number of endmembers , L number of band, h,w of hyperspectral image
# beta two encoder x reg loss , delta two encoder reg loss , gamma two encoder ANS reg loss
# sparsed_decay low rank for A loss
# weight_decay_param
#LR1 to M, LR2 to A

#--------------------init dataset and hyperparameter--------
if dataset == 'samson':
    #0-a:0.81
    #1-a:0.73
    #2-a:0.92
    #3-a:0.94
    seed = 1
    image_file = r'./data/samson_dataset.mat'
    P, L, col = 3, 156, 95
    LR1,LR2, EPOCH, batch_size = 1e-3,1e-3, 300,95*95 #1e-3,1e-3, 100,2500 #1e-3,1e-3, 200,2500
    beta, delta, gamma = 2e+1, 0, 0.91
    sparse_decay, weight_decay_param = 2e-6, 2e-4
    drop = 0.1
    D_model = 16 #16
    D_state = 8
    D_conv = 4
    T_MAX = 40
    J=6
    # sparse_decay, weight_decay_param = 1e-4, 1e-4
    index = [2,1,0]
    init_E =True#True 
    train_display = True #True
    NonZeroClipper_FLAG = False#False
    lr_scheduler_FLAG = False#True
    clip_grad = True #True
    unemb = 'max' #| 'mean' | linear
    emaabu = False

elif dataset == 'jasper': 
    ##0-a:0.96
    #1-a:0.95
    #2-a:0.99
    #3-a:0.97
    seed = 888
    image_file = r'./data/jasper_dataset.mat'
    P, L, col = 4, 198, 100
    LR1,LR2, EPOCH, batch_size=3e-3,3e-3,200,7500 #1e-3,1e-4,300,7500   #3e-3,3e-3,200,7500
    beta, delta, gamma =1, 0,0.99                 # 0.90                #0.97
    sparse_decay, weight_decay_param = 2e-6, 2e-4  # #2e-4
    drop = 0.2 #0.2
    D_model =8#32 #16
    D_state =8#8 
    D_conv = 4
    J=6
    T_MAX =300#BEST 250
    index = [3,1,2,0]  
    init_E = True
    train_display = True
    NonZeroClipper_FLAG = False
    lr_scheduler_FLAG = True #True
    clip_grad = True #False
    unemb = 'max' #'max'| 'mean' | linear
    emaabu = True

elif dataset == 'urban':
    #0-a:0.96
    #1-a:0.77
    #2-a:0.94
    #3-a:0.88
    seed = 888
    image_file = r'./data/urban_dataset.mat'
    P, L, col = 4, 162, 307
    LR1,LR2, EPOCH, batch_size = 3e-2,3e-3,200,20000 #3e-2,3e-3,200,20000#8e-3,8e-4,300,20000
    beta, delta, gamma = 1, 0,0.98
    sparse_decay, weight_decay_param =  2e-6,2e-4 #2e-6,2e-4 
    drop = 0.2 
    D_model =8#8
    D_state = 8
    D_conv = 4
    T_MAX =200#150 #200 #300
    J=6 
    index = [3,1,2,0]  
    init_E = True
    train_display = True
    NonZeroClipper_FLAG = False
    lr_scheduler_FLAG = True   #True #True  
    clip_grad = False # False
    unemb = 'max' #max      #| 'mean' | linear  | max
    emaabu = True


else:
    raise ValueError("Unknown dataset")

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
 
    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
 
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
 
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True,warn_only=False)

    

set_seed(seed)






#--------------init model--------
# if model_name == 'CycUnet':
#     net = CycUnet(L,P,drop).to(device)
# elif model_name == 'DAEN':
#     net = DAEN(L,P,drop).to(device)
# elif model_name == 'DConvAEN':
#     net = DConvAEN(L,P,drop).to(device)
# elif model_name == 'CMUNet':
#     net = CMUNet(L,P,drop,D_state).to(device)
# elif model_name == 'CMUNet_se':
#     net = CMUNet(L,P,drop,D_model,D_state,D_conv,J,unemb).to(device)
# elif model_name == 'CMUNet_se':
if model_name == 'ProMU':
    net = ProMU(L,P,drop,D_model,D_state,D_conv,J,unemb).to(device)
    
#--------------------------------
print(net)
#--------------init Endmember--------
if dataset == 'samson':
    data = sio.loadmat(image_file)
    Y = torch.from_numpy(minmax_scale(data['Y'])).float() 
    A = torch.from_numpy(data['A']).float() 
    M_true = data['M'] 
    E_VCA,_,_ = vca(data['Y'],P)
    print(E_VCA.shape)
    E_VCA_init = torch.from_numpy(minmax_scale(E_VCA)).float()
    print(E_VCA_init.shape)
    Y=torch.reshape(Y,(L,col,col))
    A=torch.reshape(A,(P,col,col))

if dataset == 'jasper':
    data = sio.loadmat(image_file)
    Y = torch.from_numpy((data['Y'])).float() 
    A = torch.from_numpy(data['A']).float() 
    M_true = data['M'] 
    E_VCA,_,_ = vca((data['Y']),P)
    print(E_VCA.shape)
    E_VCA_init = torch.from_numpy(data['M1']).float() 
    print(E_VCA_init.shape)
    Y=torch.reshape(Y,(L,col,col))
    A=torch.reshape(A,(P,col,col))

if dataset == 'urban':
    data = sio.loadmat(image_file)
    Y = torch.from_numpy(minmax_scale(data['Y'])).float() 
    A = torch.from_numpy(data['A']).float() 
    M_true = data['M'] #
    E_VCA,_,_ = vca(data['Y'],P)
    print(E_VCA.shape)
    E_VCA_init = torch.from_numpy(minmax_scale(E_VCA)).float()
    print(E_VCA_init.shape)
    Y=torch.reshape(Y,(L,col,col))
    A=torch.reshape(A,(P,col,col))

# if dataset == 'dc':
#     data = sio.loadmat(image_file)
#     Y = torch.from_numpy(minmax_scale(data['Y'])).float() 
#     A = torch.from_numpy(data['S_GT'].reshape(290*290,-1).T).float() 
#     M_true = data['GT'].T 
#     _,index_vca,_ = vca(data['Y'],P)
#     print(index_vca)
#     E_VCA = ((data['Y']))[:,index_vca]
#     print(E_VCA.shape)
#     E_VCA_init = torch.from_numpy((E_VCA)).float()
#     print(E_VCA_init.shape)
#     Y=torch.reshape(Y,(L,col,col))
#     A=torch.reshape(A,(P,col,col))
    
# if dataset == 'apex':
#     data = sio.loadmat(image_file)
#     Y = torch.from_numpy((data['Y'])).float()  
#     A = torch.from_numpy(data['A']).float()  
#     M_true = data['M'] 
#     E_VCA,_,_ = vca((data['Y']),P)
#     print(E_VCA.shape)
#     E_VCA_init = torch.from_numpy((data['M1'])).float() 
#     print(E_VCA_init.shape)
#     Y=torch.reshape(Y,(L,col,col))
#     A=torch.reshape(A,(P,col,col))

#Define Dataset
class MyTrainData(torch.utils.data.Dataset):
  """
input:
  eg:
    img: torch.Size([n_band, n_row,n_col])
    gt: torch.Size([n_endmembers, n_row,n_col])
    transform: None
return:
    x: torch.Size([156, 95, 95])
    y: torch.Size([3, 95, 95])

  """
  def __init__(self, img, gt, transform=None):
    #img.shape torch.Size([156, 95, 95])  gt.shape torch.Size([3, 95, 95])
    n_band,n_row,n_col = img.shape
    self.img_all = img.float()
    # make the img shape to (n_row*n_col, n_band,1)
    img = img.reshape(n_band, n_row*n_col).transpose(1,0)
    gt = gt.reshape(P, n_row*n_col).transpose(1,0)
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img[idx,:],self.gt[idx,:] 

  def __len__(self):
    return self.img.shape[0]
  def get_x_all(self):
    return self.img



class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6,1.0)

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma_reg=gamma):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input).to(device)
        loss = self.loss(input, target_tensor) / input.size(0)
        return gamma_reg*loss


if model_name == 'ProMU':
    def weights_init(m):
        
        nn.init.kaiming_normal_(net.conv1d.weight.data)
        # nn.init.kaiming_normal_(net.conv1d2.weight.data)
        nn.init.kaiming_normal_(net.SIR0.conv1d1.weight.data)
        nn.init.kaiming_normal_(net.SIR0.conv1d2.weight.data)
        # nn.init.kaiming_normal_(net.bandlayer0.efc3.weight.data)
        nn.init.kaiming_normal_(net.SIR1.conv1d1.weight.data)
        nn.init.kaiming_normal_(net.SIR1.conv1d2.weight.data)
        # torch.nn.init.constant_(net.bandlayer0.SIR.proj.weight.data,1. / (D_conv*p))
        nn.init.kaiming_normal_(net.SIR2.conv1d1.weight.data)
        nn.init.kaiming_normal_(net.SIR2.conv1d2.weight.data)
        # nn.init.kaiming_normal_(net.bandlayer1.efc3.weight.data)
        nn.init.kaiming_normal_(net.SIR3.conv1d1.weight.data)
        nn.init.kaiming_normal_(net.SIR3.conv1d2.weight.data)
        nn.init.kaiming_normal_(net.SIR4.conv1d1.weight.data)
        nn.init.kaiming_normal_(net.SIR4.conv1d2.weight.data)
        # torch.nn.init.constant_(net.bandlayer1.SIR.proj.weight.data,1. / (D_conv*p))
        # nn.init.kaiming_normal_(net.MioA_auto[1].adpstate.adaptive.weight.data)
        # nn.init.kaiming_normal_(net.adpstate.linear.weight.data)
        nn.init.kaiming_normal_(net.mambablock0.Process_aware_Branch.efc1.weight.data)
        nn.init.kaiming_normal_(net.mambablock0.Process_aware_Branch.efc2.weight.data)
        nn.init.kaiming_normal_(net.mambablock1.Process_aware_Branch.efc1.weight.data)
        nn.init.kaiming_normal_(net.mambablock1.Process_aware_Branch.efc2.weight.data)
        nn.init.kaiming_normal_(net.mambablock2.Process_aware_Branch.efc1.weight.data)
        nn.init.kaiming_normal_(net.mambablock2.Process_aware_Branch.efc2.weight.data)
        nn.init.kaiming_normal_(net.mambablock3.Process_aware_Branch.efc1.weight.data)
        nn.init.kaiming_normal_(net.mambablock3.Process_aware_Branch.efc2.weight.data)
        nn.init.kaiming_normal_(net.mambablock4.Process_aware_Branch.efc1.weight.data)
        nn.init.kaiming_normal_(net.mambablock4.Process_aware_Branch.efc2.weight.data)
        nn.init.kaiming_normal_(net.MioA_auto[0].Process_aware_Branch.efc1.weight.data)
        nn.init.kaiming_normal_(net.MioA_auto[0].Process_aware_Branch.efc2.weight.data)


train_dataset= MyTrainData(img=Y,gt=A, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           )
eva_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,drop_last=False,)

# net=AutoEncoder()
net.apply(weights_init)
criterionSumToOne = SumToOneLoss()
model_dict = net.state_dict()
if init_E and model_name == 'CycUnet':
    model_dict['decoder1.0.weight'] = E_VCA_init
    model_dict['decoder2.0.weight'] = E_VCA_init
elif init_E and model_name != 'CycUnet':
    model_dict['decoder.0.weight'] = E_VCA_init
net.load_state_dict(model_dict)
print('**************plot init endmembers********************')
#get pred endmembers 
M_pred_init = model_dict['decoder.0.weight'].data.cpu().numpy()[:,index]
plt.figure()
# 使用 plot 函数绘制光谱
plt.plot(minmax_scale(M_pred_init[:, :]),color='red')
plt.plot(minmax_scale(M_true[:, :]),color='blue')
esam = eSAM(M_pred_init, M_true, L, P)
print('VCA sad:{:.5f}'.format(esam))
# cbar_pred = plt.colorbar(im_pred)
# 显示图形
plt.savefig('初始endmembers.png')
plt.close()
def nuclear_norm_batch(inputs):
    summed = torch.sum(inputs, dim=0)  
    reshaped = summed.view(-1, summed.size(-1)) 
    norm = torch.norm(reshaped, p='nuc')
    return norm
#-----------------init loss func----------
if loss == 'MSE':
    def loss_func(x, x_bar,epoch):
        loss = F.mse_loss(x, x_bar,reduction='sum') / x.shape[0]
        return loss
elif loss == 'SAD':
    def loss_func(x,x_bar,epoch):
        SAD_loss = SAD(L)
        return SAD_loss(x,x_bar).sum() / x.shape[0]
elif loss == 'MSE_SAD':
    def loss_func(x,x_bar,epoch,beta):
        sad = SAD(L) #->(b,1,1)
        SAD_loss = sad(x,x_bar).sum() / x.shape[0]
        MSE_loss = F.mse_loss(x, x_bar,reduction='mean')
        # print(SAD_loss.data.cpu()) 
        # print(MSE_loss.data.cpu())
        return  SAD_loss  +  beta*MSE_loss

elif loss == 'SID':
    def loss_func(x,x_bar,epoch):
        SID_loss = SID()
        return SID_loss(x,x_bar)

elif loss == 'top_K_SAD_MSE':
    def gettop_k_percent(epoch):
        # top_k_percent =math.exp(-(epoch / EPOCH))

        top_k_percent = 1- (epoch / EPOCH ) **0.2
        # if top_k_percent < 0.8:
        #     top_k_percent = 0.8
        return top_k_percent


    def loss_func(x,x_bar,epoch):
        sad = SAD(L) #->(b,1,1)
        SAD_loss = sad(x,x_bar)
        MSE_loss = F.mse_loss(x, x_bar,reduction='none').sum(-1) 
        loss_sort_sad,_ =torch.sort(SAD_loss,dim=0,descending=True)
        loss_sort_mse,_ = torch.sort(MSE_loss,dim=0,descending=True)
        top_k_percent = gettop_k_percent(epoch)
        top_K = int(top_k_percent * x.shape[0])
        SAD_loss = torch.sum(loss_sort_sad[0:top_K,:,:]) /top_K
        MSE_loss = torch.sum(loss_sort_mse[0:top_K]) /top_K 
        return 0.5 * SAD_loss + 0.5 * MSE_loss



    
params_to_optimize = [
    {'params': param, 'lr': LR1 if 'decoder' in name else LR2 }
    for name, param in net.named_parameters()
]
optimizer = torch.optim.Adam(params_to_optimize,weight_decay=weight_decay_param)
ema_model = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(gamma))

if lr_scheduler_FLAG == True:
     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_MAX, T_mult=2)
apply_clamp_inst1 = NonZeroClipper()
#log hp
print(f'modelname:{model_name},dataset:{dataset},\
      loss:{loss},seed:{seed},lr1:{LR1},lr2:{LR2},t_max:{T_MAX},,EPOCH:{EPOCH},\
        batch_size:{batch_size},drop:{drop},D_model:{D_model},D_state:{D_state},\
        D_conv:{D_conv},init_E:{init_E},NonZeroClipper_FLAG:{NonZeroClipper_FLAG},\
                lr_scheduler_FLAG:{lr_scheduler_FLAG}, clip_grad:{clip_grad},mode_unemb:{unemb},spare_decay:{sparse_decay},gamma:{gamma}')
save_dir = os.path.join(dataset, model_name)
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist
time_start = time.time()
for epoch in range(EPOCH):
    for i, (x,_) in enumerate(train_loader):
        net.train()
        x = x.to(device)
        abu_est1,re_result,x_emb = net(x)
        loss_sumtoone = criterionSumToOne(abu_est1)
        loss_re = loss_func(re_result,x,epoch,beta)
        loss_nuc = nuclear_norm_batch(x_emb)
        total_loss =loss_re+loss_sumtoone + sparse_decay * loss_nuc
        optimizer.zero_grad()
        total_loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
        optimizer.step()
        ema_model.update_parameters(net)
        if NonZeroClipper_FLAG:
            net.decoder.apply(apply_clamp_inst1)
        print('Epoch:', epoch, '| i:', i,'| train loss: %.4f' % total_loss.data.cpu().numpy(),'| sumtoone loss: %.4f' % loss_sumtoone.data.cpu().numpy(),'| re loss: %.4f' % loss_re.data.cpu().numpy())
    if lr_scheduler_FLAG == True:
        scheduler.step()
    #画图
    if (epoch+1 )% 10 == 0 :
        all_abu_est = []  
        all_re = []  

        for j, (x, _) in enumerate(eva_loader):
            net.eval()  
            with torch.no_grad():
                x = x.to(device)
                if emaabu:
                    abu_est1, re_result1,_ =ema_model(x) 
                else:
                    abu_est1, re_result1,_ =net(x)  
            all_abu_est.append(abu_est1.detach().cpu().numpy()) 
            all_re.append(re_result1.detach().cpu().numpy())
        abu_est1 = np.concatenate(all_abu_est, axis=0).T.reshape(P,col,col)  
        re_result1 = np.concatenate(all_re, axis=0)
        abu_est1 = abu_est1[index,:,:]
        A1 = A
        A1 = A1.detach().data.cpu().numpy()
        Y1 = Y.detach().data.cpu().numpy()
        x =  train_dataset.get_x_all()
        e_pred = (ema_model.module.decoder[0].weight.data.cpu().numpy())[:,index] 
        e_true = M_true 
        armse = aRMSE(A1, abu_est1, P, col)
        esam = eSAM(e_true, e_pred, L, P)
        rsam = SAM(x.transpose(1, 0), re_result1.transpose(1, 0), L, col)
       

        print('**********************************')
        print('aRMSE: {:.5f}'.format(armse) + \
            ' | eSAM:{:.5f}'.format(esam) + \
                ' | rSAM:{:.5f}'.format(rsam))
        # print('**********************************')
        # print('#0-b:{:.2f}'.format(ema_model.module.mambablock0.Process_aware_Branch.adpfactor.data.cpu().numpy()))
        # print('#1-b:{:.2f}'.format(ema_model.module.mambablock1.Process_aware_Branch.adpfactor.data.cpu().numpy()))
        # print('#2-b:{:.2f}'.format(ema_model.module.mambablock2.Process_aware_Branch.adpfactor.data.cpu().numpy()))
        # print('#3-b:{:.2f}'.format(ema_model.module.mambablock3.Process_aware_Branch.adpfactor.data.cpu().numpy()))
        
        # print('#0-cs:{:.2f}'.format(ema_model.module.SIR0.adpfactor.data.cpu().numpy()))
        # print('#1-cs:{:.2f}'.format(ema_model.module.SIR1.adpfactor.data.cpu().numpy()))
        # print('#2-cs:{:.2f}'.format(ema_model.module.SIR2.adpfactor.data.cpu().numpy()))
        # print('#3-cs:{:.2f}'.format(ema_model.module.SIR3.adpfactor.data.cpu().numpy()))
        e_pred = minmax_scale(e_pred)
        e_true = minmax_scale(e_true)
        fig = plt.figure(figsize=(22, 12))  
        gs = fig.add_gridspec(3, P + 1, height_ratios=[1, 1, 0.4], hspace=0.02, wspace=0.1) 
        for m in range(P):
            ax = fig.add_subplot(gs[0, m])
            im = ax.imshow(abu_est1[m, :, :], cmap='jet', vmin=0, vmax=1)  
            ax.set_title(f"pred {m + 1}", fontsize=12)  
            ax.set_axis_off()  
        for m in range(P):
            ax = fig.add_subplot(gs[1, m])
            ax.imshow(A1[m, :, :], cmap='jet', vmin=0, vmax=1)  
            ax.set_title(f"gt {m + 1}", fontsize=12)
            ax.set_axis_off()  

        
        cax = fig.add_axes([0.92, 0.2, 0.02, 0.6]) 
        cbar = fig.colorbar(im, cax=cax)

        for m in range(P):
            ax = fig.add_subplot(gs[2, m])
            ax.plot(e_pred[:, m], label=f"Endmember {m + 1} (Pred)", color='blue') 
            ax.plot(e_true[:, m], label=f"Endmember {m + 1} (True)", color='orange') 
            ax.set_title(f"Endmember {m + 1}", fontsize=12)
            ax.set_xlabel("L", fontsize=10)
            ax.set_ylabel("Value", fontsize=10) 
        plt.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05)
        save_png_path = os.path.join(save_dir, '丰度与端元.png')
        plt.savefig(save_png_path)
        plt.close()
        save_mat_path = os.path.join(save_dir, 'final_results.mat')
        sio.savemat(save_mat_path, {'Y': Y1, 'abu_est': abu_est1, 'A': A1, 'M': M_true, 'M_pred': minmax_scale(e_pred)})
    
time_end = time.time()
print('total computational cost:', time_end-time_start)

