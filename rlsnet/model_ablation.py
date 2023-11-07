"""该版本用384*640的图像, 用于论文的消融实验"""
# import sys
# sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlsnet.backbone import *
from rlsnet.decoder import BUSD

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super(non_bottleneck_1d, self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
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
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureAggregationModule(torch.nn.Module):
    """cat两个相同尺寸的张量, 然后将通道数缩减至1/4,再执行两次注意力机制"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels//4

        self.convblock = conv_bn_relu(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1) #torch.Size([1, 256, 24, 40])
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x) # torch.Size([1, 64, 24, 40])
        x1 = self.avgpool(feature) # torch.Size([1, 64, 1, 1])
        attention_1 = self.sigmoid(self.relu(self.conv1(x1)))
        x1 = torch.mul(feature, attention_1) #torch.Size([1, 64, 24, 40])

        x2 = self.maxpool(x1) # torch.Size([1, 64, 1, 1])
        attention_2 = self.sigmoid(self.bn(self.conv2(x2))) # torch.Size([1, 64, 1, 1])
        x2 = torch.mul(x1, attention_2) # torch.Size([1, 64, 24, 40])
        result = torch.add(x2, feature) # torch.Size([1, 64, 24, 40])
        return result

class RLSNet(torch.nn.Module):
    """RLSnet整体结构"""
    def __init__(self, args, pretrained=True):
        super(RLSNet, self).__init__()
        self.args = args
        
        if args.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']:
            self.model = my_resnet(args.backbone, pretrained=pretrained, use_BFA=args.use_BFA)
        else:
            print("please check your model!!!")

        """特征聚合-baseline"""
        self.seg_up4 = conv_bn_relu(512, 512//4, 1) if args.backbone in ['34','18'] else conv_bn_relu(2048, 2048//4, 1)
        self.seg_up3 = conv_bn_relu(256, 256//4, 1) if args.backbone in ['34','18'] else conv_bn_relu(1024, 1024//4, 1)
        self.seg_up2 = conv_bn_relu(128, 128//4, 1) if args.backbone in ['34','18'] else conv_bn_relu( 512,  512//4, 1)
        self.seg_up1 = conv_bn_relu(64,   64//4, 1) if args.backbone in ['34','18'] else conv_bn_relu( 256,  256//4, 1)

        self.seg_px3 = conv_bn_relu(256, 256//2, 1) if args.backbone in ['34','18'] else conv_bn_relu(1024, 1024//2, 1)
        self.seg_px2 = conv_bn_relu(128, 128//2, 1) if args.backbone in ['34','18'] else conv_bn_relu( 512,  512//2, 1)
        self.seg_px1 = conv_bn_relu(64 ,  64//2, 1) if args.backbone in ['34','18'] else conv_bn_relu( 256,  256//2, 1)

        """特征聚合-FAM"""
        self.FAM34 = FeatureAggregationModule(in_channels = 256) if args.backbone in ['34','18'] else FeatureAggregationModule(in_channels = 1024)
        self.FAM23 = FeatureAggregationModule(in_channels = 128) if args.backbone in ['34','18'] else FeatureAggregationModule(in_channels = 512)
        self.FAM12 = FeatureAggregationModule(in_channels =  64) if args.backbone in ['34','18'] else FeatureAggregationModule(in_channels = 256)

        """可行驶区域"""
        self.combine_drivable = nn.Sequential(
                conv_bn_relu( 16, 128, 3, padding=2, dilation=2) if args.backbone in ['34','18'] else conv_bn_relu(64, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                nn.Conv2d(128, args.road_classes, 1))

        """车道线"""
        self.combine_lane = nn.Sequential(
                conv_bn_relu( 32, 128, 3, padding=2, dilation=2) if args.backbone in ['34','18'] else conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4))
        self.lane_conv = nn.Conv2d(128, args.lane_classes, 1)
        self.lane_decoder = BUSD(img_height = 384, img_width = 640, num_classes = args.lane_classes)
        
        """场景识别-baseline"""
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.scenes_classes) if args.backbone in ['34','18'] else nn.Linear(2048, args.scenes_classes) 

        """场景识别-SCM"""
        self.road_layers = nn.Sequential(
                        nn.Conv2d(512, 64, (3, 3), stride=2, padding=2) if args.backbone in ['34','18'] else nn.Conv2d(2048, 256, (3, 3), stride=2, padding=2),
                        nn.MaxPool2d(2, stride=2),
                        nn.BatchNorm2d(64, eps=1e-3) if args.backbone in ['34','18'] else nn.BatchNorm2d(256, eps=1e-3),
                        non_bottleneck_1d(64, 0.3, 1) if args.backbone in ['34','18'] else non_bottleneck_1d(256, 0.3, 1))
        self.linear_1 = nn.Linear(960, 256) if args.backbone in ['34','18'] else nn.Linear(3840, 1024) #取决于图片的大小
        self.linear_2 = nn.Linear(256, args.scenes_classes) if args.backbone in ['34','18'] else nn.Linear(1024, args.scenes_classes)

    def forward(self, x):
        bs = x.size()[0]
        x1, x2, x3, fea = self.model(x)
        #x1.shape = [1, 64, 96, 160]
        #x2.shape = [1, 128, 48, 80]
        #x3.shape = [1, 256, 24, 40]
        #fea.shape = [1, 512, 12, 20]
        """
        语义特征融合
        """
        if self.args.use_FAM:
            x4 = self.seg_up4(fea) # torch.Size([1, 128, 12, 20])
            x4_ = nn.functional.interpolate(x4, size=(24, 40), mode='bilinear',align_corners=False) # torch.Size([1, 128, 24, 40])
            x3 = self.seg_px3(x3) # torch.Size([1, 128, 24, 40])
            x34 = self.FAM34(x4_, x3) # torch.Size([1, 64, 24, 40])
            x34_ = nn.functional.interpolate(x34, size=(48, 80), mode='bilinear',align_corners=False) # torch.Size([1, 64, 48, 80])

            x2 = self.seg_px2(x2) # torch.Size([1, 64, 48, 80])
            x234 = self.FAM23(x34_, x2) # torch.Size([1, 32, 48, 80])
            x234_ = nn.functional.interpolate(x234, size=(96, 160), mode='bilinear',align_corners=False) # torch.Size([1, 32, 96, 160])

            x1 = self.seg_px1(x1) # torch.Size([1, 32, 96, 160])
            x1234 = self.FAM12(x234_, x1) # torch.Size([1, 16, 96, 160])

        else:
            x4 = self.seg_up4(fea) # torch.Size([2, 128, 12, 20])
            x4_ = nn.functional.interpolate(x4, size=(24, 40), mode='bilinear',align_corners=False) # torch.Size([1, 128, 24, 40])
            x3 = self.seg_px3(x3) # torch.Size([2, 128, 24, 40])
            x34 = torch.cat([x4_, x3],dim=1) # torch.Size([1, 256, 24, 40])
            x34 = self.seg_up3(x34) # torch.Size([2, 64, 24, 40])
            x34_ = nn.functional.interpolate(x34, size=(48, 80), mode='bilinear',align_corners=False) # torch.Size([1, 64, 48, 80])
            
            x2 = self.seg_px2(x2) # torch.Size([1, 64, 48, 80])
            x234 = torch.cat([x34_, x2],dim=1) # torch.Size([1, 128, 48, 80])
            x234 = self.seg_up2(x234)  # torch.Size([1, 32, 48, 80])
            x234_ = nn.functional.interpolate(x234, size=(96, 160), mode='bilinear',align_corners=False) # torch.Size([1, 32, 96, 160])
            
            x1 = self.seg_px1(x1) # torch.Size([1, 32, 96, 160])
            x1234 = torch.cat([x234_, x1],dim=1) # torch.Size([1, 64, 96, 160])
            x1234 = self.seg_up1(x1234) # torch.Size([1, 16, 96, 160])

        seg_drivable = self.combine_drivable(x1234) # 输出128, 96, 160
        seg_lane = self.combine_lane(x234) # 输出128, 90, 160 torch.Size([2, 128, 48, 80])

        """可行驶区域"""
        out_drivable = nn.functional.interpolate(seg_drivable, size=(384, 640), mode='bilinear', align_corners=False)
        
        """车道线"""
        if self.args.use_BUSD:
            out_lane = self.lane_decoder(seg_lane)
        else:
            out_lane = self.lane_conv(seg_lane)
            out_lane = nn.functional.interpolate(out_lane, size=(384, 640), mode='bilinear', align_corners=False)
        
        """场景辨识"""
        if self.args.use_SCM:
            output_scenes = self.road_layers(fea).view(bs, -1) # torch.Size([2, 960])
            output_scenes = self.linear_2(self.linear_1(output_scenes)) # 
        else:
            output_scenes = self.avgpool(fea)
            output_scenes = torch.flatten(output_scenes, 1)
            output_scenes = self.fc(output_scenes)

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
    args.backbone = '18'
    args.lane_classes = 11
    args.road_classes = 3
    args.scenes_classes = 4
    args.use_psa = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ## 先把图片放到cuda上，再把模型放到cuda上
    # b, c, h, w
    img = torch.rand(1, 3, 384, 640).cuda()

    model = RLSNet(args)
    model = model.cuda()

    out_drivable, out_lane, output_scenes = model(img)
    print(out_drivable.size())
    print(out_lane.size())
    print(output_scenes.size())