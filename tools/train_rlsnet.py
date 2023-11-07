"""
RLSNet在bdd100k数据集上训练, 包含场景识别分支、车道线检测、可行驶区域辨识
"""

import sys 
sys.path.append('./')
import argparse
from torch.utils.data import DataLoader
# from rlsnet.model_v1 import RLSNet
from rlsnet.model_v1 import RLSNet
import os
import torch
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from util.loss import DiceLoss, OhemCrossEntropy
from dataset import bdd100k


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


def train(args, model, optimizer, dataloader_train, dataloader_val):
    log_result = open(args.save_model_path + 'log_result.txt', 'a')
    log_result.write(str(args)+'\n'+'\n')

    Epoch_result = open(args.save_model_path + 'Epoch_result.txt', 'a')
    Epoch_result.write(args.note+'\n'+'\n')
    Epoch_result.write("Epoch, Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable, Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, Acc_road, loss_train_mean\n")

    drivable_loss = torch.nn.CrossEntropyLoss().cuda()
    lane_loss = OhemCrossEntropy(ignore_label = args.ignore_label).cuda()
    scenes_func = torch.nn.CrossEntropyLoss().cuda()

    step, it = 0, 0
    iters = len(dataloader_train)
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()

        loss_record = []
        for sample, scenes_id in dataloader_train:
            data, drivable_label, lane_label = sample['image'], sample['drivable_label'], sample['lane_label'] #torch.Size([1, 720, 1280]) #torch.Size([1])
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                drivable_label = drivable_label.cuda()
                lane_label = lane_label.cuda()
                scenes_label = scenes_id.cuda()

            out_drivable, out_lane, output_scenes = model(data)
            
            """计算各项任务的损失"""
            loss_drivable = drivable_loss(out_drivable, drivable_label.long())
            loss_lane = lane_loss(out_lane, lane_label.long())
            loss_scenes = scenes_func(output_scenes, scenes_label.long())

            loss = loss_drivable + loss_lane + 0.1*loss_scenes
            loss = loss_lane

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

            if step == 0 or step % 10 == 0:
                msg = 'Epoch: [{}/{}] Iter:[{}/{}] lr: {}, totle_loss: {:.6f}, drivable loss: {:.6f}, lane loss: {:.6f}, scenes loss: {:.6f}'.format( \
                       epoch, args.num_epochs, it, iters, lr, loss, loss_drivable, loss_lane, loss_scenes)
                print(msg)
                log_result.write(msg+'\n')
            
            if it == iters:
                it = 0  
            else:
                it+=1

            step += 1

        loss_train_mean = np.mean(loss_record)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 :
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, str(epoch)+'.pth'))

        if epoch % args.validation_step == 0:
            Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,\
            Acc_pix_lane,     Acc_class_lane,     mIoU_lane,     FWIoU_lane, \
            Acc_road = val(args, epoch, model, dataloader_val)

            Epoch_result.write(str(epoch)+', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
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
    parser.add_argument('--num_epochs', type=int, default = 100, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="BDD100K", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default = 720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default = 1280, help='Width of cropped/resized input image to network')
    parser.add_argument('--ratio', type=int, default=1, help='ratio')
    parser.add_argument('--num_workers', type=int, default=6, help='ratio')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone', type=str, default="resnet50",help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--road_classes', type=int, default = 3, help='num of workers')
    parser.add_argument('--lane_classes', type=int, default=11, help='num of object classes (with void)')
    parser.add_argument('--scenes_classes', type=int, default=4, help='num of object classes (with void)')
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

    # 1、创建结果路径
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    
    #2、读取数据
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    dataloader_train, dataloader_val = make_data_loader(args, **kwargs)

    # 3、建立模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model = RLSNet(args)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # torch.cuda.set_device(1)
    # model.cuda()
    args.learning_rate = args.learning_rate/4*args.batch_size

    # 4、建立优化器
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # 5、开始训练
    train(args, model, optimizer, dataloader_train, dataloader_val)


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
        '--save_model_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.17)model_v1_only_lane/',
        '--backbone', '50',  # only support resnet50 and resnet101
        '--optimizer', 'sgd',
        '--ignore_label', '255',
        '--note', '用的v1, 只预测可行驶区域'
    ]
    main(params)