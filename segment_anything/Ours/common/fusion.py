"""
https://github.com/med-air/CMC#
"""
import torch
from torch import nn
from torch.nn import init, Conv3d
import torch.nn.functional as F



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
def gcd(a, b):  # Greatest Common Divisor
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x, groups):
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, depth, height, width)
    return x


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
    


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm3d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])



    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)

            if self.dw_parallel == False:
                x = x + dw_out

        # You can return outputs based on what you intend to do with them
        return outputs



class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride, 
                 kernel_sizes=[1,3,5], 
                 expansion_factor=2, 
                 dw_parallel=True, 
                 add=True, 
                 activation='relu6'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv3d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )

        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales

        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv3d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm3d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv3d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
    

    def forward(self, x):
        # [B, 768, 8, 8, 8]
        pout1 = self.pconv1(x)  # [1, 4608, 8, 8, 8]
        msdc_outs = self.msdc(pout1)  # len = 3, each shape is [B, 4608, 8, 8, 8]

        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout += dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)

        # gcd(4608, 768)  --> [B, 4608, 8, 8, 8]
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))  

        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        


# def MSCBLayer(in_channels, 
#               out_channels, n=1, 
#               stride=1, 
#               kernel_sizes=[1,3,5], 
#               expansion_factor=2, 
#               dw_parallel=True, 
#               add=True, 
#               activation='relu6'):
#     convs = []
#     mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#     convs.append(mscb)
#     if n > 1:
#         for i in range(1, n):
#             mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
#             convs.append(mscb)
#     conv = nn.Sequential(*convs)
#     return conv


class MSCBLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 n=1, 
                 stride=1, 
                 kernel_sizes=[1,3,5], 
                 expansion_factor=2, 
                 dw_parallel=True, 
                 add=True, 
                 activation='relu6'):
        super().__init__()

        layers = []
        layers.append(
            MSCB(in_channels, out_channels, stride,
                 kernel_sizes=kernel_sizes,
                 expansion_factor=expansion_factor,
                 dw_parallel=dw_parallel,
                 add=add,
                 activation=activation)
        )

        for _ in range(1, n):
            layers.append(
                MSCB(out_channels, 
                out_channels, 
                     1,
                     kernel_sizes=kernel_sizes,
                     expansion_factor=expansion_factor,
                     dw_parallel=dw_parallel,
                     add=add,
                     activation=activation)
            )

        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)
        

class MultiScaleConvBlock(nn.Module):
    def __init__(self, 
                 in_channels=768, 
                 out_channels=768, 
                 mscb_layers=2, 
                 stride=1, 
                 kernel_sizes=[1,3,5], 
                 expansion_factor=6, 
                 dw_parallel=True, 
                 add=True, 
                 activation='relu6'):
        super().__init__()
        self.cab = CAB(in_channels=in_channels)
        self.sab = SAB()
        self.mscb = MSCBLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            n=mscb_layers, 
            stride=stride, 
            kernel_sizes=kernel_sizes, 
            expansion_factor=expansion_factor, 
            dw_parallel=dw_parallel, 
            add=add, 
            activation=activation
        )

    def forward(self, x):
        out = self.cab(x) * x
        out = self.sab(out) * x
        out = self.mscb(out)
        return out
    

# class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
#     def _check_input_dim(self, input):

#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
#         #super(ContBatchNorm3d, self)._check_input_dim(input)

#     def forward(self, input):
#         self._check_input_dim(input)
#         return F.batch_norm(
#             input, self.running_mean, self.running_var, self.weight, self.bias,
#             True, self.momentum, self.eps)
    

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU(inplace=True)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out
    

class FusionLayer(nn.Module):
    def __init__(self, in_channel=384, outChans=384, act='relu'):  # 512, 512, 1, act='relu'
        super().__init__()
        self.layer1 = LUConv(in_channel * 3, (in_channel * 3 + outChans)//2, act)
        self.layer2 = LUConv((in_channel * 3 + outChans)//2, outChans,act)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2, x3):
        concat = torch.cat([x1, x2, x3], dim=1)  # Concatenate along channel dimension
        cov_layer1 = self.layer1(concat)
        cov_layer2 = self.layer2(cov_layer1)
        return self.sigmoid(cov_layer2)  # Element-wise multiplication with the concatenated input



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
    t2w_embedding = torch.randn(1, 384 , 8 , 8, 8).to(device)

    # net_1 = CAB(in_channels=768).to(device)
    # net_2 = SAB().to(device)
    # net_3 = MSCBLayer(in_channels=768, out_channels=768, 
    #                   n=2, stride=1, kernel_sizes=[1,3,5], 
    #                   expansion_factor=6, dw_parallel=True, add=True, 
    #                   activation='relu6').to(device)
    
    # output = net_1(t2w_embedding) * t2w_embedding
    # output = net_2(output) * t2w_embedding
    # output = net_3(output)

    # fusion_block = MultiScaleConvBlock(in_channels=768, out_channels=768, 
    #                                   mscb_layers=2).to(device)

    fusion_block = FusionLayer().to(device)

    print(fusion_block(t2w_embedding, t2w_embedding, t2w_embedding).shape)

    # print(t2w_embedding.shape)
    # adc_embedding = torch.randn(1, 768, 8, 8, 8)
    # dwi_embedding = torch.randn(1, 768, 8, 8, 8)

    # out = net(t2w_embedding)
    # print(out.shape)
    # print(net.parameters())
    # # 统计网络的参数量
    print(sum(p.numel() for p in fusion_block.parameters() if p.requires_grad))
