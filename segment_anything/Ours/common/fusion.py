"""
https://github.com/med-air/CMC#
"""
import torch
from torch import nn
from torch.nn import init


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




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    net = MIA_MOdule(in_dim=768).to(device)
    t2w_embedding = torch.randn(1, 768, 8, 8, 8).to(device)
    print(t2w_embedding.shape)
    # adc_embedding = torch.randn(1, 768, 8, 8, 8)
    # dwi_embedding = torch.randn(1, 768, 8, 8, 8)

    # out = net(t2w_embedding)
    # print(out.shape)
    # print(net.parameters())
    # # 统计网络的参数量
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))
