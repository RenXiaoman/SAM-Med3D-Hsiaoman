"""
https://github.com/med-air/CMC#
"""
import torch
from torch import nn
from torch.nn import init, Conv3d


class SEAttention3D(nn.Module):
    def __init__(self, channel=384, reduction=16):
        super().__init__()
        # 对 D × H × W 做全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 两层全连接，先降维再升维，最后得到 3C
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 3 * channel, bias=False),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3):
        # 输入: [B, C, D, H, W]
        B, C, D, H, W = x1.size()
        # 融合
        x = x1 + x2 + x3
        # Squeeze: [B, C, D, H, W] -> [B, C, 1, 1, 1] -> [B, C]
        y = self.avg_pool(x).view(B, C)
        # Excitation: [B, C] -> [B, 3C] -> [B, 3C, 1, 1, 1]
        y = self.fc(y).view(B, 3 * C, 1, 1, 1)
        # split
        weight1 = torch.sigmoid(y[:, :C, :, :, :])
        weight2 = torch.sigmoid(y[:, C:2*C, :, :, :])
        weight3 = torch.sigmoid(y[:, 2*C:, :, :, :])
        # scale & fuse
        out = x1 * weight1 + x2 * weight2 + x3 * weight3
        return out


#################################################################
class MIA_MOdule(nn.Module):
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.channel_dim = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        B, C, D, H, W = x.size()
        q = x.view(B, C, -1)
        k = x.view(B, C, -1).permute(0, 2, 1)
        v = x.view(B, C, -1)

        qk = q @ k
        qk_new = torch.max(qk, -1, keepdim=True)[0].expand_as(qk) - qk
        attention = self.softmax(qk_new)

        out = attention @ v
        out = out.view(B, C, D, H, W)
        return self.gamma * out + x


#################################################################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


# Other types of layers can go here (e.g., nn.Linear, etc.)


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        assert (in_channels // ratio), 'in_channels must be divisible by ratio'
        self.reduce_channels = in_channels // ratio

        self.glob_avg = nn.AdaptiveAvgPool3d(1)
        self.glob_max = nn.AdaptiveMaxPool3d(1)
        # self.glob_max = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(self.in_channels, self.reduce_channels, 1, bias=False),
            act_layer(activation, inplace=True),
            nn.Conv3d(self.reduce_channels, self.out_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):  # [B, 768, 8, 8, 8]
        B, C, _, _, _ = x.size()
        avg_pool_out = self.glob_avg(x)  # [1, 768, 1, 1, 1]
        avg_out = self.fc(avg_pool_out)  # [1, 768]

        max_pool_out = self.glob_max(x)
        max_out = self.fc(max_pool_out)

        assert (avg_out.shape == max_out.shape), 'avg_out and max_out must have the same shape'

        return self.sigmoid(avg_out + max_out)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = Conv3d(2, 1, kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # x [B, C, D, H, W]
        max_result, _ = torch.max(x, dim=1, keepdim=True)   # [B, 1, D, H, W]
        avg_result = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        result = torch.cat([max_result, avg_result], 1)  # [B, 2, D, H, W]
        output = self.conv(result)  # # [B, 1, D, H, W]
        output = self.sigmoid(output)  # 通过sigmoid获得权重:(B,1,H,W)
        return output


class EMCA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
    net_1 = CAB(in_channels=768).to(device)
    net_2 = SAB().to(device)
    t2w_embedding = torch.randn(1, 768 , 8 , 8, 8).to(device)
    output = net_1(t2w_embedding) * t2w_embedding
    output = net_2(output) * t2w_embedding + t2w_embedding
    print(output.shape)

    # print(t2w_embedding.shape)
    # adc_embedding = torch.randn(1, 768, 8, 8, 8)
    # dwi_embedding = torch.randn(1, 768, 8, 8, 8)

    # out = net(t2w_embedding)
    # print(out.shape)
    # print(net.parameters())
    # # 统计网络的参数量
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))
