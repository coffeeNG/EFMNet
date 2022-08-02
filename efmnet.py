import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import arch_util as arch_util

class FA(nn.Module):
    def __init__(self, nf):
        super(FA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out
    
class FAComponent(nn.Module):
    def __init__(self, nf, k_size=3):
        super(FAComponent, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 1)
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv1(x))
        out = torch.mul(self.conv3(x), y) 
        return out

class LAcomponent(nn.Module):
    def __init__(self, in_dim):
        super(LAcomponent, self).__init__()
        self.chanel_in = in_dim
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x
        
class EFMmodule(nn.Module):
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(EFMmodule, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv2_a = nn.Conv2d(group_width, group_width, kernel_size=3, padding=1, dilation=1, bias=False)

        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv2_b = nn.Conv2d(group_width, group_width, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv3_b = nn.Conv2d(group_width, group_width, kernel_size=3, padding=1, dilation=1, bias=False)

        self.conv1_c = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv2_c = nn.Conv2d(group_width, group_width, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv3_c = nn.Conv2d(group_width, group_width, kernel_size=3, padding=2, dilation=2, bias=False)

        self.conv1_d = nn.Conv2d(nf, nf, kernel_size=3, padding=1, dilation=1, bias=False)

        self.LAcomponent = LAcomponent(nf)
        self.FAcomponent = FAComponent(nf)
        
        self.convall = nn.Conv2d(group_width * 3, nf, kernel_size=1, bias=False)
        self.convall_ = nn.Conv2d(nf*2, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out_a = self.conv1_a(x)
        out_a = self.lrelu(out_a)
        out_a = self.conv2_a(out_a)

        out_b = self.conv1_b(x)
        out_b = self.conv2_b(self.lrelu(out_b))
        out_b = self.conv3_b(self.lrelu(out_b))

        out_c = self.conv1_c(x) 
        out_c = self.conv2_c(self.lrelu(out_c)) 
        out_c = self.conv3_c(self.lrelu(out_c))
        
        out_d = self.conv1_d(x)

        out = self.convall(torch.cat([out_a, out_b, out_c], dim=1))
        out += out_d

        branch_cs = self.LAcomponent(out)
        branch_pa = self.FAcomponent(out)
        out = self.convall_(torch.cat([branch_cs, branch_pa], dim=1))

        return out
    
class EFMNet(nn.Module):
    
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(EFMNet, self).__init__()
        EFM = functools.partial(EFMmodule, nf=nf, reduction=2)
        self.scale = scale
        
        self.sfe = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        self.FTM = arch_util.make_layer(EFM, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = FA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = FA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        
        fea = self.sfe(x)
        fea = self.trunk_conv(self.FTM(fea))   
        
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out