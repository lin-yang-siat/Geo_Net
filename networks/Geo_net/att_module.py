###########################################################################


import numpy as np
import torch
import math
import torch.nn as nn

torch_ver = torch.__version__[:3]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAM_Module(nn.Module):
    """Distance Attention Module (DAM)"""

    def __init__(self, in_dim):
        """
        Initializes the Distance Attention Module.

        Parameters:
            in_dim (int): The input feature map's channel dimension.
        """
        super(DAM_Module, self).__init__()
        self.chanel_in = in_dim

        # Define 1x1 convolutions for query, key, and value projections
        self.query_conv = nn.Conv3d(in_dim, 1, kernel_size=1)
        self.key_conv = nn.Conv3d(in_dim, 1, kernel_size=1)

        # Value convolution with input as 3 times the input channels
        self.value_conv = nn.Conv3d(in_channels=3 * in_dim, out_channels=in_dim, kernel_size=1)

        # Initialize learnable parameters gamma for scaling
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.gamma_g = nn.Parameter(torch.zeros(1))

        # Softmax for attention weight normalization
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Forward pass for the Distance Attention Module.

        Parameters:
            q (Tensor): Query feature maps (B x C x L x H x W)
            k (Tensor): Key feature maps (B x C x L x H x W)
            v (Tensor): Value feature maps (B x C x L x H x W)

        Returns:
            Tensor: The attended output feature map, after applying attention and a residual connection.
            Tensor: The attention map (B x (HxW) x (HxW))
        """
        # Get the batch size, channels, and spatial dimensions
        B, C, L, H, W = q.size()

        # Reshape the query to apply the convolution
        q = q.reshape(-1, C, L, H, W)
        proj_query = self.query_conv(q)
        proj_query = proj_query.view(B, -1, L * W * H).permute(0, 2, 1)

        # Reshape the key to apply the convolution
        k = k.reshape(-1, C, L, H, W)
        proj_key = self.key_conv(k)
        proj_key = proj_key.view(B, -1, L * W * H)

        # Compute the attention scores using the query and key
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Reshape the value feature maps and concatenate the input feature maps
        value = v.reshape(-1, C, L, H, W)
        value = torch.cat((value, q, k), dim=1)

        # Apply the value convolution to obtain the attended value
        proj_value_x = self.value_conv(value)
        proj_value_x = proj_value_x.view(B, -1, L * W * H)

        # Compute the attended output by multiplying with the attention weights
        out_x = torch.bmm(proj_value_x, attention.permute(0, 2, 1))
        out_x = out_x.view(B, C, L, H, W)

        # Apply the residual connection with a learned scaling factor (gamma_x)
        out_x = self.gamma_x * out_x + v

        return out_x
