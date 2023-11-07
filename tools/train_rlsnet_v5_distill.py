"""
RLSNet在bdd100k数据集上训练, 包含场景识别分支、车道线检测、可行驶区域辨识,利用deeplab v3进行知识蒸馏
"""

from email.mime import image
import sys 
sys.path.append('./')
import argparse
from torch.utils.data import DataLoader
from rlsnet.model_v5 import RLSNet
import os
import torch
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from util.loss import DiceLoss, OhemCrossEntropy
from utils import create_logger
from dataset import bdd100k
import pprint
import torch.backends.cudnn as cudnn
from configs import config
from configs import update_config
from deeplabv3.deeplab import *

class CriterionPixelWise(nn.Module):
    def __init__(self, loss_weight=0.1):
        super(CriterionPixelWise, self).__init__()
        self.loss_weight = loss_weight
        # self.ignore_index = ignore_index
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss*self.loss_weight

class CriterionPixelWiseLossLogits(nn.Module):
    """ Logits pixel wise loss calculation module.
    Args:
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        loss_weight (float, optional): Weight of loss.Defaults to 1.0.
    """
    def __init__(self, tau=1.0, loss_weight=1.0):
        super(CriterionPixelWiseLossLogits, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward function."""
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1, W*H)/self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum( - softmax_pred_T * logsoftmax(preds_S.view(-1,W*H)/self.tau))
        return self.loss_weight*loss / (C * N)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def val(args, epoch, model, dataloader):

    print('start val!')
    model.eval()
    tbar = tqdm.tqdm(dataloader, desc='\r')
    test_loss = 0.0

    drivable_evaluator = Evaluator(args.road_classes)
    drivable_evaluator.reset()

    lane_evaluator = Evaluator(args.lane_classes)
    lane_evaluator.reset()

    running_corrects = 0

    for i, (sample, scenes_id) in enumerate(tbar):
        images, drivable_targets, lane_targets = sample['image'], sample['drivable_label'], sample['lane_label']
        if args.use_gpu:
            images, drivable_targets, lane_targets, scenes_label = images.cuda(), drivable_targets.cuda(),lane_targets.cuda(), scenes_id.cuda()
        with torch.no_grad():
            out_drivable, out_lane, output_scenes = model(images)
        
        """场景识别评估"""
        _, scenes_pred = torch.max(output_scenes, 1)
        running_corrects += torch.sum(scenes_pred == scenes_label.data)

        """可行驶区域评估"""
        drivable_preds = out_drivable.data.cpu().numpy()
        drivable_preds = np.argmax(drivable_preds, axis=1)
        drivable_targets = drivable_targets.cpu().numpy()
        drivable_evaluator.add_batch(drivable_targets, drivable_preds)

        """车道线评估"""
        lane_preds = out_lane.data.cpu().numpy()
        lane_preds = np.argmax(lane_preds, axis=1)
        lane_targets = lane_targets.cpu().numpy()
        lane_evaluator.add_batch(lane_targets, lane_preds)
        
    
    """可行驶区域评估"""
    Acc_pix_drivable = drivable_evaluator.Pixel_Accuracy()
    Acc_class_drivable = drivable_evaluator.Pixel_Accuracy_Class()
    mIoU_drivable = drivable_evaluator.Mean_Intersection_over_Union()
    FWIoU_drivable = drivable_evaluator.Frequency_Weighted_Intersection_over_Union()
    
    """车道线评估"""
    Acc_pix_lane = lane_evaluator.Pixel_Accuracy()
    Acc_class_lane = lane_evaluator.Pixel_Accuracy_Class()
    mIoU_lane = lane_evaluator.Mean_Intersection_over_Union()
    FWIoU_lane = lane_evaluator.Frequency_Weighted_Intersection_over_Union()

    """场景识别评估"""
    Acc_road = running_corrects.double() / (i+1)
    Acc_road = Acc_road.cpu().numpy()

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, i+1))
    print("Driving area Segment: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {} \n \
        Lane line Segment: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {} \n \
            scenes classify: Acc_road: {} ".format(Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,
                Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, 
                Acc_road)
                )
    print('Loss: %.3f' % test_loss)

    return Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,\
            Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, \
            Acc_road

def train(args, student_model, teacher_model, optimizer, dataloader_train, dataloader_val, logger, final_output_dir):
    drivable_loss = torch.nn.CrossEntropyLoss().cuda()
    lane_loss = OhemCrossEntropy(ignore_label = args.ignore_label).cuda()
    scenes_func = torch.nn.CrossEntropyLoss().cuda()
    distill_loss = CriterionPixelWiseLossLogits(loss_weight= 0.1)

    step, it = 0, 0
    iters = len(dataloader_train)
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        student_model.train()
        teacher_model.eval()

        loss_record = []
        for sample, scenes_id in dataloader_train:
            images, drivable_label, lane_label = sample['image'], sample['drivable_label'], sample['lane_label'] #torch.Size([1, 720, 1280]) #torch.Size([1])
            if torch.cuda.is_available() and args.use_gpu:
                images = images.cuda()
                drivable_label = drivable_label.cuda()
                lane_label = lane_label.cuda()
                scenes_label = scenes_id.cuda()

            out_drivable, out_lane, output_scenes = student_model(images)
            with torch.no_grad():
                out_dri = teacher_model(images)
            
            """计算各项任务的损失"""
            loss_drivable = drivable_loss(out_drivable, drivable_label.long())
            loss_lane = lane_loss(out_lane, lane_label.long())
            loss_scenes = scenes_func(output_scenes, scenes_label.long())
            d_loss = distill_loss(out_drivable, out_dri)

            loss = loss_drivable + loss_lane + 0.1*loss_scenes + d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

            if step == 0 or step % 10 == 0:
                msg = 'Epoch: [{}/{}] Iter:[{}/{}] lr: {}, totle_loss: {:.6f}, drivable loss: {:.6f}, lane loss: {:.6f}, scenes loss: {:.6f}'.format( \
                       epoch, args.num_epochs, it, iters, lr, loss, loss_drivable, loss_lane, loss_scenes)
                # print(msg)
                logger.info(msg)
            
            if it == iters-1:
                it = 0  
            else:
                it+=1

            step += 1

        loss_train_mean = np.mean(loss_record)
        # print('loss for train : %f' % (loss_train_mean))
        logger.info('loss for train : %f' % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 :
            if not os.path.isdir(final_output_dir):
                os.mkdir(final_output_dir)
            torch.save(student_model.module.state_dict(), os.path.join(final_output_dir, str(epoch)+'.pth'))

        if epoch % args.validation_step == 0:
            Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,\
            Acc_pix_lane,     Acc_class_lane,     mIoU_lane,     FWIoU_lane, \
            Acc_road = val(args, epoch, student_model, dataloader_val)
            
            logger.info("Epoch, Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable, Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, Acc_road, loss_train_mean")
            logger.info("head:"+str(epoch)+', {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,\
                Acc_pix_lane,     Acc_class_lane,     mIoU_lane,     FWIoU_lane, \
                Acc_road,         loss_train_mean)+'\n')

def make_data_loader(args, **kwargs):
    train_set = bdd100k.BDD100kSegmentation(args, split='train')
    val_set = bdd100k.BDD100kSegmentation(args, split='val')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
    return train_loader, val_loader

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default = 50, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="BDD100K", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default = 720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default = 1280, help='Width of cropped/resized input image to network')
    parser.add_argument('--ratio', type=int, default=1, help='ratio')
    parser.add_argument('--num_workers', type=int, default=6, help='ratio')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone', type=str, default="50",help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--deeplab_backbone', type=str, default="resnet",help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--norm', type=str, default='gn',choices=['gn', 'bn', 'abn'],help='normalization methods')
    parser.add_argument('--road_classes', type=int, default = 3, help='num of workers')
    parser.add_argument('--lane_classes', type=int, default=10, help='num of object classes (with void)')
    parser.add_argument('--scenes_classes', type=int, default=4, help='num of object classes (with void)')
    parser.add_argument('--out-stride', type=int, default=16,help='network output stride (default: 8)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--ignore_label', type=int, default=255, help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--note', type=str, default='', help='path to save model')
    parser.add_argument('--use_psa', type=bool, default=True, help='path to save model')
    args = parser.parse_args(params)

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    #2、读取数据
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    dataloader_train, dataloader_val = make_data_loader(args, **kwargs)

    # 3、建立模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    student_model = RLSNet(args)
    teacher_model = DeepLab(args=args, num_classes=3)
    teacher_model.load_state_dict(torch.load('/workspace/xiaozhihao/RLSnet/resnet_pretrained_pth/resnet101.pth',map_location='cpu')['state_dict'], strict=False)
    
    logger, final_output_dir= create_logger(args, 'train')
    logger.info(pprint.pformat(args))

    if torch.cuda.is_available() and args.use_gpu:
        student_model = torch.nn.DataParallel(student_model).cuda()
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    # torch.cuda.set_device(1)
    # model.cuda()
    args.learning_rate = args.learning_rate/4*args.batch_size

    # 4、建立优化器
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(student_model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(student_model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # 5、开始训练
    train(args, student_model, teacher_model, optimizer, dataloader_train, dataloader_val, logger, final_output_dir)

if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '0.01',
        '--data', '/workspace/xiaozhihao/RLSnet/BDD100K',
        '--num_workers', '12',
        '--road_classes', '3',
        '--lane_classes', '10',
        '--scenes_classes', '4',
        '--cuda', '0', 
        '--batch_size', '1',  # 6 for resnet101, 12 for resnet18
        '--save_model_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.18)model_v5/',
        '--backbone', '50',  # only support resnet50 and resnet101
        '--optimizer', 'sgd',
        '--ignore_label', '255',
        '--note', 'v5_data_aug_deeplab'
    ]
    main(params)