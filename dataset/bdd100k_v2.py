import os
import sys 
sys.path.append('./')
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
import util.custom_transforms as tr
import json
import cv2
from dataset.base_dataset import BaseDataset
import tqdm
"""该份代码用于生成数据的字典，格式为：
database = [
0:{'image':path
        'lane_label':path
        'drivable_label':path},
1:{'image':path
        'lane_label':path
        'drivable_label':path},
2:     
]

"""


class BDD100kDataset(BaseDataset):
    def __init__(self, args, split):
        super().__init__(args, split)
        
        self.files = {}
        self.files[split] = [line.strip() for line in open('/workspace/xiaozhihao/RLSnet/BDD100K/'+self.split+'_rlsnet.txt')]
        self.db = self._get_db()
    
    def _get_db(self):
        print("Found %d %s images" % (len(self.files[self.split]), self.split))
        print('building database...')
        gt_db = {}
        for i, file_name in enumerate(self.files[self.split]):
            img_path = os.path.join(self.images_base, file_name + '.jpg')
            lane_label_path = os.path.join(self.lane_seg, file_name + '.png')
            drivable_label_path = os.path.join(self.drivable_seg, file_name + '_drivable_id.png')
            json_label_path = os.path.join(self.json_base, file_name + '.json')
            gt_db[i] = {
                'image': img_path,
                'lane_label': lane_label_path,
                'drivable_label': drivable_label_path, 
                'json_label': json_label_path
                }
        print('database build finish')
        return gt_db

    def __len__(self):
        return len(self.files[self.split])

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 720
    args.crop_size = 720
    args.ratio = 1
    args.data = '/workspace/xiaozhihao/BiSeNet/BDD100K'

    bdd100k_train = BDD100kDataset(args, split='train')
