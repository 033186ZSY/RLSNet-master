import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):                                 # x  torch.Size([1, 64, 96, 160])
        input_x = self.conv_v_right(x)                   # input_x  torch.Size([1, 32, 96, 160])

        batch, channel, height, width = input_x.size() 
        input_x = input_x.view(batch, channel, height * width)    # torch.Size([1, 32, 15360])
        context_mask = self.conv_q_right(x)# [N, 1, H, W]         # torch.Size([1, 1, 96, 160])
        context_mask = context_mask.view(batch, 1, height * width)# torch.Size([1, 1, 15360])
        context_mask = self.softmax_right(context_mask)           # torch.Size([1, 1, 15360])

        context = torch.matmul(input_x, context_mask.transpose(1,2))#torch.Size([1, 32, 1])
        context = context.unsqueeze(-1)                            # torch.Size([1, 32, 1, 1])
        context = self.conv_up(context)                            # torch.Size([1, 64, 1, 1])
        mask_ch = self.sigmoid(context)                            # torch.Size([1, 64, 1, 1])
        out = x * mask_ch                                          # torch.Size([1, 64, 96, 160])

        return out

    def channel_pool(self, x):
        
        g_x = self.conv_q_left(x)                                             # torch.Size([1, 32, 96, 160])
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)                                            # torch.Size([1, 32, 1, 1])

        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)      # torch.Size([1, 1, 32])
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)# torch.Size([1, 32, 15360])
        
        context = torch.matmul(avg_x, theta_x)                                      # torch.Size([1, 1, 15360])
        context = self.softmax_left(context)                                        # torch.Size([1, 1, 15360])
        context = context.view(batch, 1, height, width)                             # torch.Size([1, 1, 96, 160])
        mask_sp = self.sigmoid(context)                                             # torch.Size([1, 1, 96, 160])
        out = x * mask_sp                                                           # torch.Size([1, 64, 96, 160])            
        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x) # torch.Size([1, 64, 96, 160])
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()
        input_x = input_x.view(batch, channel, height * width)  # [N, IC, H*W]
        context_mask = self.conv_q_right(x)  # [N, 1, H, W]
        context_mask = context_mask.view(batch, 1, height * width)  # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)  # [N, 1, H*W]
        context = torch.matmul(input_x, context_mask.transpose(1, 2))  # [N, IC, 1]
        context = context.unsqueeze(-1)  # [N, IC, 1, 1]
        context = self.conv_up(context) # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)  # [N, OC, 1, 1]
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        
        g_x = self.conv_q_left(x)  # [N, IC, H, W]
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)  # [N, IC, 1, 1]
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)  # [N, 1, IC]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width) # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)  # [N, IC, H*W]
        context = torch.matmul(avg_x, theta_x)  # [N, 1, H*W]
        context = context.view(batch, 1, height, width) # [N, 1, H, W]
        mask_sp = self.sigmoid(context)  # [N, 1, H, W]
        out = x * mask_sp
        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)
        # [N, C, H, W]
        out = self.channel_pool(out)
        # [N, C, H, W]
        # out = context_spatial + context_channel
        return out