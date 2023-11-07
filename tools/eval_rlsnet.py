"""
此代码用于评估RLSnet网络在bdd100k数据上的精度
"""
import sys 
sys.path.append('./')
from rlsnet.model_ablation import RLSNet
import torch
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
# from util import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
import tqdm
from dataset import bdd100k_v2
from util.utils import DataLoaderX


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

def eval(args, model, dataloader, txt):
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
    print('numImages: %5d]' % (i+1))
    print("Driving area Segment: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {} \n \
              Lane line Segment: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {} \n \
                scenes classify: Acc_road: {} ".format(Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,
                Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, 
                Acc_road)
                )
    print('Loss: %.3f' % test_loss)
    txt.write(str(Acc_pix_drivable) + ', ' +
            str(Acc_class_drivable) + ', ' +
            str(mIoU_drivable) + ', ' +
            str(FWIoU_drivable) + ', ' +
            str(Acc_pix_lane) + ', ' +
            str(Acc_class_lane) + ', ' +
            str(mIoU_lane) + ', ' +
            str(FWIoU_lane) + ', ' +
            str(Acc_road) 
    )
    return Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,\
            Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, \
            Acc_road

def make_data_loader(args, **kwargs):
    # train_set = bdd100k_v2.BDD100kDataset(args, split='train')
    val_set = bdd100k_v2.BDD100kDataset(args, split='val')
    # train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoaderX(val_set, batch_size=1, shuffle=False, **kwargs)
    # return train_loader, val_loader
    return val_loader

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="BDD100K", help='Dataset you are using.')
    parser.add_argument('--height', type=int, default = 384, help='Height of cropped/resized input image to network')
    parser.add_argument('--width', type=int, default = 640, help='Width of cropped/resized input image to network')
    parser.add_argument('--ratio', type=int, default=1, help='ratio')
    parser.add_argument('--num_workers', type=int, default=6, help='ratio')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone', type=str, default="18",help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--road_classes', type=int, default=3, help='num of workers')
    parser.add_argument('--lane_classes', type=int, default=10, help='num of object classes (with void)')
    parser.add_argument('--scenes_classes', type=int, default=4, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--ignore_label', type=int, default=255, help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--note', type=str, default='', help='path to save model')
    parser.add_argument('--drivable_only', action='store_true', default=False, help='path to save model')
    parser.add_argument('--lane_only', action='store_true', default=False, help='path to save model')
    parser.add_argument('--scenes_only', action='store_true', default=False, help='path to save model')
    parser.add_argument('--whole', action='store_true', default=False, help='path to save model')

    parser.add_argument('--use_BFA', type=bool, default=False, help='是否使用BFA')
    parser.add_argument('--use_SCM', type=bool, default=False, help='是否使用SCM')
    parser.add_argument('--use_FAM', type=bool, default=False, help='是否使用FAM')
    parser.add_argument('--use_BUSD', type=bool, default=False, help='是否使用BUSD')
    args = parser.parse_args(params)

    # create dataset and dataloader
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    kwargs = {'num_workers': 4, 'pin_memory': False}
    dataloader_val = make_data_loader(args, **kwargs)

    # build model
    model = RLSNet(args)
    model.cuda()
    # load pretrained model if exists


    for file in os.listdir(args.checkpoint_path):
        txt = open(args.save_model_path + 'val_result.txt', 'a')
        txt.write(args.note+'\n'+'\n')
        txt.write("Epoch, Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable, Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, Acc_road"+'\n')

        if file.endswith('.pth'):
            print('load model from %s ...' % args.checkpoint_path)
            print(file)
            model.load_state_dict(torch.load(args.checkpoint_path+file))
            print('Done!')
            txt.write(file.split('.')[0]+', ')
            # test
            eval(args, model, dataloader_val, txt)


if __name__ == '__main__': 

    params = [
        '--checkpoint_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.12.17)Ablation/baseline_1218/',
        '--save_model_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.12.17)Ablation/baseline_1218/',
        '--data', '/workspace/xiaozhihao/BiSeNet/BDD100K',
        '--road_classes', '3',
        '--lane_classes', '10',
        '--scenes_classes', '4',
        '--cuda', '1', 
        '--backbone', '18',  # only support resnet50 and resnet101
        # '--note', '(2022.10.11)RLS_resnet50模型评估（不做后处理）'
        '--whole',
    ]
    main(params)