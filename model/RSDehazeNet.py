import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# channel refinement block
class CRB(nn.Module):
    def __init__(self,in_channels,r):
        super(CRB,self).__init__()
        self.branch_layer=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=in_channels//r,
                                                  kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=in_channels//r,
                                                  out_channels=in_channels,
                                                  kernel_size=1),
                                        nn.Sigmoid())
    def forward(self,x):
        xb=self.branch_layer(x)
        return x*xb

# residual channel refinement block
class RCRB(nn.Module):
    def __init__(self,r):
        super(RCRB,self).__init__()
        self.banch_layer=nn.Sequential(nn.Conv2d(in_channels=32,
                                                 out_channels=32,
                                                 kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=32,
                                                 out_channels=32,
                                                 kernel_size=3,
                                                 padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=32,
                                                 out_channels=32,
                                                 kernel_size=1),
                                       CRB(in_channels=32,
                                           r=r))
    def forward(self,x):
        xb=self.banch_layer(x)
        return x+xb

# 3 residual channel refinement blocks
class TRCRB(nn.Module):
    def __init__(self,r):
        super(TRCRB,self).__init__()
        self.b1=RCRB(r=r)
        self.b2=RCRB(r=r)
        self.b3=RCRB(r=r)
    def forward(self,x):
        x1=self.b1(x)
        x2=self.b2(x1)
        x3=self.b3(x2)
        return x1,x2,x3
        
# feature fusion block
class FFB(nn.Module):
    def __init__(self):
        super(FFB,self).__init__()
        self.layer=nn.Sequential(nn.Conv2d(in_channels=96*4,
                                           out_channels=96,
                                           kernel_size=1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=96,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1),
                                 nn.ReLU())
    def forward(self,x1,x2,x3,x4):
        x=torch.concat([x1,x2,x3,x4])
        x=self.layer(x)
        return x

# 
