import torch
import torchvision
import torch.nn.modules
from rlsnet.resnet_BFA import resnet18, resnet34, resnet50, resnet101

class my_resnet(torch.nn.Module):
    def __init__(self,layers, pretrained = True, use_BFA = False):
        super(my_resnet,self).__init__()
        if layers == '18' and not use_BFA:
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '18' and use_BFA:
            model = resnet18(pretrained=pretrained)

        elif layers == '34' and not use_BFA:
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '34' and use_BFA:
            model = resnet34(pretrained=pretrained)

        elif layers == '50' and not use_BFA:
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '50' and use_BFA:
            model = resnet50(pretrained=pretrained)

        elif layers == '101' and not use_BFA:
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '101' and use_BFA:
            model = resnet101(pretrained=pretrained)

        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
    def forward(self, x):
        x = self.conv1(x)  #输出144*400，64通道
        x = self.bn1(x)    #输出144*400，64通道
        x = self.relu(x)   #输出144*400，64通道
        x = self.maxpool(x)  #输出为72*200，64通道
        
        x1 = self.layer1(x)   #输出为72*200，64通道
        x2 = self.layer2(x1)  # 输出为36*100，128通道
        x3 = self.layer3(x2) # 输出为18*50，256通道
        x4 = self.layer4(x3) # 输出为9*25，512通道

        return x1, x2, x3, x4
        
if __name__ == "__main__":
    import torch
    model = my_resnet('50', pretrained=True)
    input = torch.rand(1, 3, 1280, 720)
    x2, x3, x4 = model(input)
    # print(x1.size())
    print(x2.size())
    print(x3.size())
    print(x4.size())