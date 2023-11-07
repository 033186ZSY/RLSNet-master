"""
此代码用于评估RLSnet网络在bdd100k数据上的精度
"""
import sys 
sys.path.append('./')
from rlsnet.model_v3 import RLSNet
import torch
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
# from util import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou
import tqdm
from dataset import bdd100k
from utils import colour_code_segmentation
import cv2
import copy

drivable_color_map = {'Not drivable':         [0, 0, 0], 
              'Drivable area':              [11, 11, 11], 
              'Alternative drivable area': [12, 12, 12]}

lane_color_map = {'background' :(0, 0, 0),
                'lane/double yellow_solid' :(1, 1, 1),
                  'lane/double yellow_dashed':(2, 2, 2),
                  'lane/single white_solid'  :(3, 3, 3),
                  'lane/single white_dashed' :(4, 4, 4),
                  'lane/single yellow_solid' :(5, 5, 5),
                  'lane/single yellow_dashed':(6, 6, 6),
                  'lane/double white_solid'  :(7, 7, 7),
                  'lane/double white_dashed' :(8, 8, 8),
                  'lane/road curb_solid'     :(9, 9, 9),
                  'lane/road curb_dashed'    :(10, 10, 10)}

road_ca = ['residential', 'highway', 'city street', 'other']

### 用于sum_predict的转换
look_for_drivable = {0: 0, #背景类
                    1: 1,  #实线
                    2: 2, #虚线
                    3: 1, #实线
                    4: 2, #虚线
                    5: 1, #实线
                    6: 2, #虚线
                    7: 1, #实线
                    8: 2, #虚线
                    9: 1, #实线
                    10: 2,#虚线
                    11: 8, #可行驶区域
                    12: 9 #可选行驶区域
                    }

#只保留行驶区域
post_drivable =    {0: 0, #背景类
                    0: 1,  #实线
                    0: 2,  #虚线
                    1: 8, #可行驶区域
                    1: 9 #可选行驶区域
                    }
        
def convert_label(label, seg_map, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in seg_map.items():
            label[temp == k] = v
    else:
        for k, v in seg_map.items():
            label[temp == k] = v
    return label

def drivable_post_process(drivable_predict, lane_predict):
    """利用车道线对可行驶区域进行后处理"""
    drivable_predict = cv2.cvtColor(drivable_predict, cv2.COLOR_BGR2GRAY)
    lane_predict = cv2.cvtColor(lane_predict, cv2.COLOR_BGR2GRAY)
    sum_predict = copy.deepcopy(lane_predict)
    h, w = sum_predict.shape
    for i in range(h):
        for j in range(w):
            if sum_predict[i][j] == 0:
                sum_predict[i][j] = drivable_predict[i][j] 

    sum_predict = convert_label(sum_predict, look_for_drivable, inverse=False)

    for i in range(h):
    # for i in [408,423]:
    
        line = sum_predict[i].tolist()
        pix_num = [-1]
        index = [-1]
        for j in range(w):
            if line[j] != pix_num[-1] and line[j] != 0:
                pix_num.append(line[j])
                index.append(j)

        pix_num.append(w)
        index.append(w)

        for n in range(len(pix_num)):
            n += 1
            if n+2<=len(pix_num):
                if pix_num[n: n+3] == [1, 9, 1]:
                    start = index[n+1]
                    end  = index[n+2]
                    sum_predict[i][start:end] = 0
                    n += 1

                elif pix_num[n: n+3] == [8, 1, 9]:
                    start = index[n+2]
                    end  = index[n+3]
                    sum_predict[i][start:end] = 0
                    n += 1

                elif pix_num[n: n+3] == [9, 1, 8]:
                    start = index[n]
                    end  = index[n+1]
                    sum_predict[i][start:end] = 0
                    n += 1

                elif pix_num[n: n+5] == [1, 9, 8, 9, 1]:
                    start = index[n+1]
                    end  = index[n+4]
                    sum_predict[i][start:end] = 0


                elif pix_num[n: n+4] == [1, 9, 8, 1] :
                    start = index[n+1]
                    end  = index[n+2]
                    sum_predict[i][start:end] = 0
                    if pix_num[n-1] == 8:
                        start = index[n+2]
                        end  = index[n+3]
                        sum_predict[i][start:end] = 0
                        n += 1
                    else:
                        n += 1

                elif pix_num[n: n+4] == [1, 8, 9, 1]:
                    start = index[n+2]
                    end  = index[n+3]
                    sum_predict[i][start:end] = 0
                    if pix_num[n-1] == 8:
                        start = index[n+1]
                        end  = index[n+2]
                        sum_predict[i][start:end] = 0
                        n += 1
                    else:
                        n += 1

                elif pix_num[n: n+3] == [1, 9, w]:
                    start = index[n+1]
                    sum_predict[i][start:] = 0
                    n += 1


    post_drivable_predict = convert_label(sum_predict, post_drivable, inverse=True)
    return post_drivable_predict
    
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

def eval(args, model, dataloader):
    print('start val!')
    txt = open(args.save_model_path + 'val_result.txt', 'w')
    txt.write(args.note+'\n'+'\n')
    txt.write("Acc_pix_drivable, Acc_class_drivable, mIoU_drivable, FWIoU_drivable,\
            Acc_pix_lane, Acc_class_lane, mIoU_lane, FWIoU_lane, \
            Acc_road"+'\n')

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

        """车道线评估"""
        lane_preds = out_lane.data.cpu().numpy()
        lane_preds = np.argmax(lane_preds, axis=1) # (1, 720, 1280)  max=10 min 0
        lane_targets = lane_targets.cpu().numpy()
        lane_evaluator.add_batch(lane_targets, lane_preds)

        """可行驶区域评估"""

        drivable_preds = out_drivable.data.cpu().numpy()
        drivable_preds = np.argmax(drivable_preds, axis=1) # (1, 720, 1280)  max=2 min = 1
        
        lane_for_mask = copy.deepcopy(lane_preds).squeeze()
        drivable_for_mask = copy.deepcopy(drivable_preds).squeeze()
        
        drivable_predict = colour_code_segmentation(drivable_for_mask, drivable_color_map) # 三通道
        lane_predict = colour_code_segmentation(lane_for_mask, lane_color_map) # 三通道
        drivable_predict = cv2.resize(np.uint8(drivable_predict), (1280, 720))
        lane_predict = cv2.resize(np.uint8(lane_predict), (1280, 720))

        mask = drivable_post_process(drivable_predict, lane_predict) # 返回一个一通道的mask
        h, w = mask.shape

        for i in range(h):
            for j in range(w):
                if mask[i][j] == 0:
                    drivable_preds[0][i,j] = 0

        drivable_targets = drivable_targets.cpu().numpy()
        drivable_evaluator.add_batch(drivable_targets, drivable_preds)

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
    val_set = bdd100k.BDD100kSegmentation(args, split='val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, **kwargs)
    return val_loader

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1280, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone', type=str, default="50",help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--road_classes', type=int, default = 3, help='num of workers')
    parser.add_argument('--lane_classes', type=int, default=11, help='num of object classes (with void)')
    parser.add_argument('--scenes_classes', type=int, default=4, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=3, help='num of object classes (with void)')
    parser.add_argument('--ratio', type=int, default=1, help='ratio')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--note', type=str, default='', help='path to save model')
    parser.add_argument('--use_psa', type=bool, default=True, help='path to save model')
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
        if file.endswith('.pth'):
            print('load model from %s ...' % args.checkpoint_path)
            model.load_state_dict(torch.load(args.checkpoint_path+file))
            print('Done!')

            # test
            eval(args, model, dataloader_val)


if __name__ == '__main__': 

    params = [
        '--checkpoint_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.12)RLS_resnet50/',
        '--save_model_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.11)RLS_resnet50/',
        '--data', '/workspace/xiaozhihao/BiSeNet/BDD100K',
        '--road_classes', '3',
        '--lane_classes', '11',
        '--scenes_classes', '4',
        '--cuda', '1', 
        '--backbone', '50',  # only support resnet50 and resnet101
        '--note', '(2022.10.11)RLS_resnet50模型验证'

    ]
    main(params)