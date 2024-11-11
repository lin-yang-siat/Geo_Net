###########################################################################


import numpy as np
import torch
import math
import torch.nn as nn

torch_ver = torch.__version__[:3]



class DAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(DAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_dim, 1, kernel_size=1)
        self.key_conv = nn.Conv3d(in_dim, 1, kernel_size=1)

        self.value_conv = nn.Conv3d(in_channels=3 * in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.gamma_g = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v):
        """
            inputs :
                x : input feature maps( B X C X L X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        B, C, L, H, W = q.size()
        q= q.reshape(-1,C, L, H, W)
        proj_query = self.query_conv(q)
        proj_query = proj_query.view(B, -1, L * W * H).permute(0, 2, 1)

        k = k.reshape(-1, C, L, H, W)
        proj_key = self.key_conv(k)
        proj_key = proj_key.view(B, -1, L*W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

   
        value = v.reshape(-1, C, L, H, W)
        value = torch.cat((value, q, k), dim=1)

        proj_value_x = self.value_conv(value)
        proj_value_x = proj_value_x.view(B, -1, L*W*H)
        out_x = torch.bmm(proj_value_x, attention.permute(0, 2, 1))
        out_x = out_x.view(B, C, L, H, W)

        out_x = self.gamma_x*out_x + v

        return out_x
