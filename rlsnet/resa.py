import torch.nn as nn
import torch
import torch.nn.functional as F


class RESA(nn.Module):
    def __init__(self):
        super(RESA, self).__init__()
        self.iter = 5 # 5
        chan = 128 # 128
        fea_stride = 8  # 原图相对于此时的特征图大小的倍数
        self.height = 720 // fea_stride # 46
        self.width = 1280 // fea_stride # 80
        self.alpha = 2 # 2
        conv_stride = 9 # 9

        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(chan, chan, (1, conv_stride),padding=(0, conv_stride//2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(chan, chan, (1, conv_stride),padding=(0, conv_stride//2), groups=1, bias=False)

            setattr(self, 'conv_d'+str(i), conv_vert1)
            setattr(self, 'conv_u'+str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(chan, chan, (conv_stride, 1),padding=(conv_stride//2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(chan, chan, (conv_stride, 1),padding=(conv_stride//2, 0), groups=1, bias=False)

            setattr(self, 'conv_r'+str(i), conv_hori1)
            setattr(self, 'conv_l'+str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //2**(self.iter - i)) % self.height
            setattr(self, 'idx_d'+str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //2**(self.iter - i)) % self.height
            setattr(self, 'idx_u'+str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //2**(self.iter - i)) % self.width
            setattr(self, 'idx_r'+str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //2**(self.iter - i)) % self.width
            setattr(self, 'idx_l'+str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x



if __name__ == "__main__":
    import torch
    img = torch.rand(1, 128, 90, 160).cuda()
    model = RESA().cuda()
    output = model(img)
    print(output.size)


# fea torch.Size([4, 128, 46, 80])
# resa torch.Size([4, 128, 46, 80])
# seg torch.Size([4, 7, 368, 640])
# exist torch.Size([4, 6])
