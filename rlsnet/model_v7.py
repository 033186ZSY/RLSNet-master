"""该版本用384*640的图像"""
# import sys
# sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlsnet.backbone import *
from rlsnet.decoder import BUSD, drivable_BUSD
from rlsnet.resa import RESA


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
        output = F.relu(output) # torch.Size([1, 256, 3, 5])  torch.Size([1, 512, 1, 2])
        output = self.conv1x3_1(output) # torch.Size([1, 256, 3, 5])
        output = self.bn1(output) # torch.Size([1, 256, 3, 5])
        output = F.relu(output) # torch.Size([1, 256, 3, 5])

        output = self.conv3x1_2(output) # torch.Size([1, 256, 3, 5])
        output = F.relu(output) # torch.Size([1, 256, 3, 5])
        output = self.conv1x3_2(output) # torch.Size([1, 256, 3, 5])
        output = self.bn2(output) # torch.Size([1, 256, 3, 5])

        if (self.dropout.p != 0):
            output = self.dropout(output) # torch.Size([1, 256, 3, 5])
        
        return F.relu(output+input)    #+input = identity (residual connection)

class conv_bn_relu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class xzh_up(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride=1, padding = 0, dilation=1,bias=False):
        super(xzh_up, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//4, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = nn.BatchNorm2d(in_channels//4)
        self.relu = nn.ReLU()

    def forward(self,x):
        h, w = x.size()[-2:]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if h == 23:
            x = nn.functional.interpolate(x, size=(45, 80), mode='bilinear',align_corners=False)
        else:
            x = nn.functional.interpolate(x, size=(h*2, w*2), mode='bilinear',align_corners=False)
        return x

class xzh_pingxing(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride=1, padding = 1, dilation=1, bias=False):
        super(xzh_pingxing, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size, stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = nn.BatchNorm2d(in_channels//2)
        self.relu = nn.ReLU()

    def forward(self,x):
        # h, w = x.size()[-2:]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RLSNet(torch.nn.Module):
    def __init__(self, args, pretrained=True):
        super(RLSNet, self).__init__()
        # self.training = is_training
        if args.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']:
            self.model = my_resnet(args.backbone, pretrained=pretrained)
        else:
            print("please check your model!!!")

        """
        可行驶区域与车道线分支
        """
        self.seg_up4 = xzh_up(in_channels = 512) if args.backbone in ['34','18'] else xzh_up(in_channels = 2048)
        self.seg_up3 = xzh_up(in_channels = 256) if args.backbone in ['34','18'] else xzh_up(in_channels = 1024)
        self.seg_up2 = xzh_up(in_channels = 128) if args.backbone in ['34','18'] else xzh_up(in_channels = 512)

        self.seg_px3 = xzh_pingxing(in_channels = 256) if args.backbone in ['34','18'] else xzh_pingxing(in_channels = 1024)
        self.seg_px2 = xzh_pingxing(in_channels = 128) if args.backbone in ['34','18'] else xzh_pingxing(in_channels = 512)
        self.seg_px1 = xzh_pingxing(in_channels = 64) if args.backbone in ['34','18'] else xzh_pingxing(in_channels = 256)

        self.combine_drivable = nn.Sequential(
            conv_bn_relu(64, 128, 3, padding=2, dilation=2) if args.backbone in ['34','18'] else conv_bn_relu(256, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            torch.nn.Conv2d(128, 128, 1))

        self.combine_lane = nn.Sequential(
            conv_bn_relu(128, 256, 3, padding=2, dilation=2) if args.backbone in ['34','18'] else conv_bn_relu(512, 256, 3, padding=2, dilation=2),
            conv_bn_relu(256, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            torch.nn.Conv2d(128, 128, 1))

        """场景辨识分支"""
        self.road_layers = nn.Sequential(
                        nn.Conv2d(512, 256, (3, 3), stride=2, padding=2) if args.backbone in ['34','18'] else nn.Conv2d(2048, 1024, (3, 3), stride=2, padding=2),
                        nn.MaxPool2d(2, stride=2),
                        nn.BatchNorm2d(256, eps=1e-3) if args.backbone in ['34','18'] else nn.BatchNorm2d(1024, eps=1e-3),
                        non_bottleneck_1d(256, 0.3, 1) if args.backbone in ['34','18'] else non_bottleneck_1d(1024, 0.3, 1),

                        nn.Conv2d(256, 512, (3, 3), stride=2, padding=2) if args.backbone in ['34','18'] else nn.Conv2d(1024, 2048, (3, 3), stride=2, padding=2),
                        nn.MaxPool2d(2, stride=2),
                        nn.BatchNorm2d(512, eps=1e-3) if args.backbone in ['34','18'] else nn.BatchNorm2d(2048, eps=1e-3) ,
                        non_bottleneck_1d(512, 0.3, 1) if args.backbone in ['34','18'] else non_bottleneck_1d(2048, 0.3, 1)
                        )
        self.road_linear_1 = nn.Linear(1024, 256) if args.backbone in ['34','18'] else nn.Linear(4096, 1024) #取决于图片的大小
        self.output_scenes = nn.Linear(256, args.scenes_classes) if args.backbone in ['34','18'] else nn.Linear(1024, args.scenes_classes)
       
        """其他的操作"""
        self.drivable = nn.Conv2d(128, args.road_classes, (3, 3), stride=2, padding=2)
        self.resa = RESA()
        self.lane_decoder = BUSD(img_height = 384, img_width = 640, num_classes = args.lane_classes)
        self.drivable_decoder = drivable_BUSD(img_height = 384, img_width = 640, num_classes = args.road_classes)


    def forward(self, x):
        bs = x.size()[0]
        x1, x2, x3, fea = self.model(x)
        #x.shape = [1, 3, 384, 640]
        #x1.shape = [1, 64, 96, 160]
        #x2.shape = [1, 128, 48, 80]
        #x3.shape = [1, 256, 24, 40]
        #fea.shape = ([1, 512, 12, 20]
        """
        语义特征融合
        """
        up4 = self.seg_up4(fea)
        x3 = self.seg_px3(x3)
        x34 = torch.cat([up4, x3],dim=1)

        up3 = self.seg_up3(x34)
        x2 = self.seg_px2(x2)
        x234 = torch.cat([up3, x2],dim=1)

        up2 = self.seg_up2(x234)
        x1 = self.seg_px1(x1)
        x1234 = torch.cat([up2, x1],dim=1)

        seg_drivable = self.combine_drivable(x1234) # 输出[1, 128, 96, 160]
        seg_lane = self.combine_lane(x234) # 输出[1, 128, 48, 80]

        """可行驶区域"""
        out_drivable = self.drivable(seg_drivable)
        out_drivable = nn.functional.interpolate(out_drivable, size=(384, 640), mode='bilinear', align_corners=False)
        # out_drivable = self.drivable_decoder(out_drivable)
        """车道线"""
        # seg_out = self.resa(seg_lane)
        out_lane = self.lane_decoder(seg_lane)
        
        """场景辨识"""
        output_scenes = self.road_layers(fea) # torch.Size([1, 512, 1, 2])
        output_scenes = output_scenes.view(bs, -1)  # torch.Size([1, 1024])
        output_scenes = self.road_linear_1(output_scenes) # torch.Size([1, 256])
        output_scenes = self.output_scenes(output_scenes) # torch.Size([1, 4])

        return out_drivable, out_lane, output_scenes
        # return seg_out, fea


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
    img = torch.rand(1, 3, 720, 1280).cuda()

    model = RLSNet(args)
    model = model.cuda()

    out_drivable, out_lane, output_scenes = model(img)
    print(out_drivable.size())
    print(out_lane.size())
    print(output_scenes.size())