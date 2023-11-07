"""该版本含有PSA和resa, 以及自注意力蒸馏"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rlsnet.backbone import *
from rlsnet.decoder import BUSD
from rlsnet.resa import RESA

# import sys
# sys.path.append('.')

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super(non_bottleneck_1d, self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        
    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RLSNet(torch.nn.Module):
    def __init__(self, args, pretrained=True):
        super(RLSNet, self).__init__()

        if args.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']:
            self.model = my_resnet(args.backbone, pretrained=pretrained, use_psa = args.use_psa)
        else:
            print("please check your model!!!")

        """
        可行驶区域与车道线分支
        """
        self.seg_header2 = nn.Sequential(
            conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if args.backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128,3,padding=1),
            conv_bn_relu(128, 128,3,padding=1),
            conv_bn_relu(128, 128,3,padding=1),
        )
        self.seg_header3 = nn.Sequential(
            conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if args.backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128,3,padding=1),
            conv_bn_relu(128, 128,3,padding=1),
        )
        self.seg_header4 = nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if args.backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128,3,padding=1),
        )
        self.aux_combine = nn.Sequential(
            conv_bn_relu(384, 256, 3, padding=2, dilation=2),
            conv_bn_relu(256, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            torch.nn.Conv2d(128, 128, 1)
            # output : n, num_of_lanes+1, h, w
        )

        """场景辨识分支"""
        self.road_layers = nn.Sequential(
                        nn.Conv2d(2048, 1024, (3, 3), stride=2, padding=2),
                        nn.MaxPool2d(2, stride=2),
                        nn.BatchNorm2d(1024, eps=1e-3),
                        non_bottleneck_1d(1024, 0.3, 1),

                        nn.Conv2d(1024, 2048, (3, 3), stride=2, padding=2),
                        nn.MaxPool2d(2, stride=2),
                        nn.BatchNorm2d(2048, eps=1e-3),

                        non_bottleneck_1d(2048, 0.3, 1)
                        )
        self.road_linear_1 = nn.Linear(2048 * 2 * 3, 1024)#取决图片尺寸
        self.output_scenes = nn.Linear(1024, args.scenes_classes)
       
        """其他的操作"""
        self.drivable = nn.Conv2d(128, args.road_classes, (3, 3), stride=2, padding=2)
        self.resa = RESA()
        self.lane_decoder = BUSD(img_height = 720, img_width = 1280, num_classes = args.lane_classes)



    def forward(self, x):
        bs = x.size()[0]
        x2, x3, fea = self.model(x)
        #x2.shape = [1, 512, 90, 160]
        #x3.shape = [1, 1024, 45, 80]
        #fea.shape = ([1, 2048, 23, 40]
        """
        语义分割分支
        """
        x2 = self.seg_header2(x2) # 输出 x2 = Size([1, 128, 90, 160])
        x3 = self.seg_header3(x3) 
        x3 = nn.functional.interpolate(x3, size=(90, 160), mode='bilinear',align_corners=False) #输出 x3 = Size([1, 128, 90, 160])
        x4 = self.seg_header4(fea)
        x4 = nn.functional.interpolate(x4, size=(90, 160), mode='bilinear', align_corners=False) #输出 x4 = Size([1, 128, 90, 160])
        seg = torch.cat([x2, x3, x4],dim=1) #输出 seg = Size([1, 128, 90, 160])
        seg = self.aux_combine(seg) #输出 seg = Size([1, 128, 90, 160])

        """可行驶区域"""
        out_drivable = self.drivable(seg)
        out_drivable = nn.functional.interpolate(out_drivable, size=(720, 1280), mode='bilinear', align_corners=False)
        
        """车道线"""
        seg_out = self.resa(seg)
        out_lane = self.lane_decoder(seg_out)
        
        """场景辨识"""
        output_scenes = self.road_layers(fea)
        output_scenes = output_scenes.view(bs, -1)  # torch.Size([1, 12288])
        output_scenes = self.road_linear_1(output_scenes)
        output_scenes = self.output_scenes(output_scenes)

        return out_drivable, out_lane, output_scenes
        # return seg_out, fea


if __name__ == '__main__':
    import os
    import torch
    import argparse
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ## 先把图片放到cuda上，再把模型放到cuda上
    img = torch.rand(1, 128, 90, 160).cuda()
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
    img = torch.rand(1, 3, 720, 1280).cuda()

    model = RLSNet(args)
    model = model.cuda()

    out_drivable, out_lane, output_scenes = model(img)
    print(out_drivable.size())
    print(out_lane.size())
    print(output_scenes.size())