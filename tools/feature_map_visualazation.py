"""
该代码用于将网络中的特征图可视化
注意：可视化之前需要修改网络！！！ 修改网络中的forward函数。具体修改方式如下所示。

def forward(self, x):
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    feature = self.model.layer1(x)
    x = self.model.layer2(feature)
    x = self.model.layer3(x)
    x = self.model.layer4(x)
    return feature,x
"""
import sys 
sys.path.append('./')
import argparse
from rlsnet.model_v3 import RLSNet
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def image_proprecess(img_path):
    """加载数据并预处理"""
    """针对RLSnet的图像预处理"""
    image = Image.open(img_path)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data = data_transforms(image)
    data = torch.unsqueeze(data,0)
    # print(data.shape)
    return data

def visualize_feature_map(img_batch, out_path):
    """将特征图的某一通道转化为一张图来可视化"""
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()

    channel_sum = feature_map.shape[0] # 获取通道数
    print("一共有{}通道的特征图".format(channel_sum))
    # feature_map_sum = np.expand_dims(feature_map_sum, axis=2) # 扩展维度？
    feature_map_sum  = torch.zeros(feature_map.shape[1:])
    for i in range(channel_sum):
        feature_map_split = feature_map[i,:, :]
        # feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum += feature_map_split
        print(i)
        plt.figure()
        plt.imshow(feature_map_split, cmap='gray')
        plt.axis('off')
        # plt.imsave(os.path.join(out_path, str(i)+".png"), feature_map[i])
    plt.imsave(os.path.join(out_path, str(i)+"sum.png"), feature_map_sum)


if __name__ ==  '__main__':
    image_path = "/workspace/xiaozhihao/RLSnet/BDD100K/images/test/cb38bda4-26729b52.jpg"  # 测试的图像的路径
    save_path = '/workspace/xiaozhihao/RLSnet/feature_map_visualazation'
    checkpoint_path = '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.12)RLS_resnet50（v3）/3.pth'

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
    # args.use_psa = True

    # model = models.resnet152(pretrained=True) 
    model = RLSNet(args)
    model.load_state_dict(torch.load(checkpoint_path))
    
    data = image_proprecess(image_path)
    data = data.cuda()
    model = model.cuda().eval()
    fea, seg, out_drivable = model(data)
    # Upsample = nn.Upsample(scale_factor=10, mode='bilinear')
    # output = Upsample(fea)
    # visualize_feature_map(output, save_path)

    # Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    # output = Upsample(seg)
    # visualize_feature_map(output, save_path)

    Upsample = nn.Upsample(scale_factor=6, mode='bilinear')
    output = Upsample(out_drivable)
    visualize_feature_map(output, save_path)