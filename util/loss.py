import torch.nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label = 255, thres=0.9, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = [0.4, 1.0]
        sb_weights = 1.0
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

# class OhemCELoss(nn.Module):
#     """
#     Online hard example mining cross-entropy loss:在线难样本挖掘
#     if loss[self.n_min] > self.thresh: 最少考虑 n_min 个损失最大的 pixel，
#     如果前 n_min 个损失中最小的那个的损失仍然大于设定的阈值，
#     那么取实际所有大于该阈值的元素计算损失:loss=loss[loss>thresh]。
#     否则，计算前 n_min 个损失:loss = loss[:self.n_min]
#     """
#     def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()     # 将输入的概率 转换为loss值
#         self.n_min = n_min
#         self.ignore_lb = ignore_lb
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')   #交叉熵
 
#     def forward(self, logits, labels):
#         N, C, H, W = logits.size()
#         loss = self.criteria(logits, labels).view(-1)
#         loss, _ = torch.sort(loss, descending=True)     # 排序
#         if loss[self.n_min] > self.thresh:       # 当loss大于阈值(由输入概率转换成loss阈值)的像素数量比n_min多时，取所以大于阈值的loss值
#             loss = loss[loss>self.thresh]
#         else:
#             loss = loss[:self.n_min]
#         return torch.mean(loss)
