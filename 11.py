import torch
from torch.nn.parameter import Parameter
import torch.nn as nn

class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):

        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1
        z_hat = self.bn(z)
        g = self.activation(z_hat)
        return g

    def forward(self, x):
        t = self._style_pooling(x)
        g = self._style_integration(t)
        return x * g

if __name__ == '__main__':
    import os
    import torch
    import argparse
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ## 先把图片放到cuda上，再把模型放到cuda上
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.backbone = '50'
    args.lane_classes = 11
    args.road_classes = 3
    args.scenes_classes = 4
    args.use_psa = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ## 先把图片放到cuda上，再把模型放到cuda上
    # b, c, h, w
    img = torch.rand(1, 128, 12, 20).cuda()
    output_scenes = SRMLayer(img)