import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
from networks.Geo_net.STDLayer import STDLayer3D_LIF
from networks.Geo_net.att_module import DAM_Module


class down_Plock3d(nn.Module):
    def __init__(self,input_channels: int, output_channels: int, downsample: bool = False):
        super(down_Plock3d,self).__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm3d(input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        x_conv = self.conv(x)
        if self.downsample:
            x = self.maxpool(x_conv)
        else:
            x =x_conv
        return x, x_conv
class up_conv3d(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(up_conv3d,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class Encoder(nn.Module):
    def __init__(self, input_channels: int, feature_channels: int) -> None:
        super().__init__()

        self.down1 = down_Plock3d(input_channels, feature_channels, downsample=True)
        self.down2 = down_Plock3d(feature_channels, feature_channels*2, downsample=True)
        self.down3 = down_Plock3d(feature_channels * 2, feature_channels * 4, downsample=True)
        self.down4 = down_Plock3d(feature_channels * 4, feature_channels * 8, downsample=False)


    def forward(self, x: torch.Tensor):

        x, x_conv1 = self.down1(x)  # 16
        x, x_conv2 = self.down2(x)  # 8
        x, x_conv3 = self.down3(x)
        x = self.down4(x)[0]

        return x, x_conv1, x_conv2, x_conv3

class Decoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up1 = up_conv3d(input_channels, input_channels//2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, input_channels//2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(input_channels//2),
            nn.LeakyReLU(inplace=True)
        )
        self.up2 = up_conv3d(input_channels//2, input_channels//4)
        self.conv2 = nn.Sequential(
            nn.Conv3d(input_channels//2, input_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(input_channels//4),
            nn.LeakyReLU(inplace=True)
        )
        self.up3 = up_conv3d(input_channels//4, input_channels//8)
        self.conv3 = nn.Sequential(
            nn.Conv3d(input_channels//4, input_channels // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(input_channels//8),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = nn.Conv3d(input_channels // 8, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, x_skip1: torch.Tensor, x_skip2: torch.Tensor, x_skip3: torch.Tensor) -> torch.Tensor:

        x = self.up1(x)
        x = torch.cat((x, x_skip3),dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat((x, x_skip2),dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat((x, x_skip1),dim=1)
        x = self.conv3(x)

        x = self.conv(x)

        return x

class FFN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape)==6:
            B, S, C, H, W, L = x.shape
            x = x.reshape(B * S, C,  H, W, L)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if len(x.shape) == 6:
            x = x.reshape(B, S, self.output_channels, H, W, L)

        return x


class Geo_Net(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, feature_channels: int, seq_len: int):
        super(Geo_Net, self).__init__()

        self.encoder = Encoder(input_channels, feature_channels)
        self.ffn_1 = FFN(feature_channels * 8, feature_channels * 8)
        self.decoder = Decoder(feature_channels * 8, output_channels)
        # our skip image-0
        self.encoder_P = Encoder(input_channels, feature_channels)
        self.encoder_S = Encoder(input_channels, feature_channels)
        self.ffn_S = FFN(feature_channels * 8 * seq_len, feature_channels * 8)



        self.Att1 = DAM_Module(feature_channels * 8)


        ###### skip S
        self.ffn_S_skip1 = FFN(feature_channels * seq_len, feature_channels)
        self.ffn_S_skip2 = FFN(feature_channels * seq_len*2, feature_channels*2)
        self.ffn_S_skip3 = FFN(feature_channels * seq_len*4, feature_channels*4)

        self.f_1 = FFN(feature_channels * 3, feature_channels)
        self.f_2 = FFN(feature_channels * 6, feature_channels * 2)
        self.f_3 = FFN(feature_channels * 12, feature_channels * 4)



        self.Relu = nn.LeakyReLU(inplace=True)
        self.STD = STDLayer3D_LIF(2,5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=x.unsqueeze(2)
        I = x[:,0:1,:,:,:,:]
        d_P = x[:, 1:2, :, :, :, :]
        d_S = x[:, 2:10, :, :, :, :]

        # reshape:B, S, C, H, W, L--->  B*S, C, H, W, L
        B, S, C, H, W, L = I.shape
        I = I.reshape(B * S, C, H, W, L)
        B_P, S_P, C_P, H_P, W_P, L_P = d_P.shape
        d_P = d_P.reshape(B_P * S_P, C_P, H_P, W_P, L_P)
        B_S, S_S, C_S, H_S, W_S, L_S = d_S.shape
        d_S = d_S.reshape(B_S * S_S, C_S, H_S, W_S, L_S)
        # Encoder
        f, f_skip1, f_skip2, f_skip3 = self.encoder(I)
        g_P, b_skip1, b_skip2, b_skip3 = self.encoder_P(d_P)
        g_S, p_skip1, p_skip2, p_skip3 = self.encoder_S(d_S)

        # FFN : B_S, S_S*C_S, H_S, W_S, L_S--->  B_S, C_S, H_S, W_S, L_S
        g_S = g_S.reshape(B_S, 1, S_S * g_S.shape[1], g_S.shape[2], g_S.shape[3], g_S.shape[4])
        g_S = self.ffn_S(g_S)
        g_S = F.layer_norm(g_S, g_S.shape[1:])

        # Attention
        residual = f
        f = self.Att1(f, g_P, g_S)
        f = F.layer_norm(f, f.shape[1:])
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])
        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])


        # Skip
        f_skip1 = torch.cat((f_skip1, b_skip1), dim=1)
        p_skip1 = p_skip1.reshape(B_S, S_S*p_skip1.shape[1], p_skip1.shape[2], p_skip1.shape[3], p_skip1.shape[4])  # S
        p_skip1 = self.ffn_S_skip1(p_skip1)
        f_skip1 = torch.cat((f_skip1, p_skip1), dim=1)
        f_skip1 = self.f_1(f_skip1)

        f_skip2 = torch.cat((f_skip2, b_skip2), dim=1)
        p_skip2 = p_skip2.reshape(B_S, S_S * p_skip2.shape[1], p_skip2.shape[2], p_skip2.shape[3], p_skip2.shape[4])  # S
        p_skip2 = self.ffn_S_skip2(p_skip2)
        f_skip2 = torch.cat((f_skip2, p_skip2), dim=1)
        f_skip2 = self.f_2(f_skip2)

        f_skip3 = torch.cat((f_skip3, b_skip3), dim=1)
        p_skip3 = p_skip3.reshape(B_S, S_S * p_skip3.shape[1], p_skip3.shape[2], p_skip3.shape[3], p_skip3.shape[4])  # S
        p_skip3 = self.ffn_S_skip3(p_skip3)
        f_skip3 = torch.cat((f_skip3, p_skip3), dim=1)
        f_skip3 = self.f_3(f_skip3)

        # Decoder
        x = self.decoder(f, f_skip1, f_skip2, f_skip3)


        x = self.STD(x, I)

        return x
if __name__ == "__main__":

    model = Geo_Net(1,2,16,8).cuda()
    input = torch.rand(8, 10, 48, 48, 48).cuda()  # B,L(序列),C,H,W
    output = model(input)#, input1, input2)
    print(output[:,0].mean())
    print(output[:, 1].mean())
