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


mean =[0.485, 0.456, 0.406]
std  =[0.229, 0.224, 0.225]

class BDD100kSegmentation(data.Dataset):

    def __init__(self, args, split="train"):

        self.root = args.data
        self.split = split
        self.args = args
        self.files = {}
        
        self.resize_shape = [args.height, args.width]

        self.images_base = os.path.join(self.root, 'images', self.split)
        self.drivable_seg = os.path.join(self.root, 'road_seg', self.split)
        self.lane_seg = os.path.join(self.root, 'xzh_bdd_lane', self.split)
        self.json_base = os.path.join(self.root, 'det_annotations', '100k', self.split)

        # self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')
        self.files[split] = [line.strip() for line in open('/workspace/xiaozhihao/RLSnet/BDD100K/'+self.split+'_rlsnet.txt')]
        # # REDUCING DATASET
        # if split != 'test': 
        #     self.files[split] = self.files[split][:len(self.files[split])//10]

        self.drivable_mapping = {0: 0, 
                                 1: 1, 
                                 2: 2}
        self.lane_mapping = {255: 0,
                               1: 1, 
                               2: 2, 
                               3: 3, 
                               4: 4, 
                               5: 5, 
                               6: 6, 
                               7: 7, 
                               8: 8, 
                               9: 9, 
                              10: 9}

        self.class_names = ['Not drivable', 'Drivable area', 'Alternative drivable area'] 

        self.ignore_index = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        drivable_label_path = os.path.join(self.drivable_seg, os.path.basename(img_path)[:17] + '_drivable_id.png')
        lane_label_path = os.path.join(self.lane_seg, os.path.basename(img_path)[:17] + '.png')
        json_label_path = os.path.join(self.json_base, os.path.basename(img_path)[:17] + '.json')

        _img = Image.open(self.images_base+'/'+img_path+'.jpg').convert('RGB')
        _img = _img.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)

        _name = os.path.basename(img_path)[:17] + '.png'
        with open(json_label_path) as f:
            data = json.load(f)

        road = data['attributes']['scene']
        road_map = ['residential', 'highway', 'city street', 'other']
        if road in road_map:
            _scenes_id = road_map.index(road)
        else:
            _scenes_id = 3

        if self.split == 'test':
            sample = {'image': _img, 'name': _name}

        elif self.args.whole:
            _tmp_drivable = cv2.imread(drivable_label_path, cv2.IMREAD_GRAYSCALE)
            _tmp_drivable = self.convert_label(_tmp_drivable, self.drivable_mapping)
            _target_drivable = Image.fromarray(_tmp_drivable)
            _target_drivable = _target_drivable.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)


            _tmp_lane = cv2.imread(lane_label_path, cv2.IMREAD_GRAYSCALE)
            _tmp_lane = self.convert_label(_tmp_lane, self.lane_mapping)
            _target_lane = Image.fromarray(_tmp_lane)
            _target_lane = _target_lane.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)


            sample = {   'image': _img, 
                'drivable_label': _target_drivable, 
                    'lane_label': _target_lane, 
                        'roadid': _scenes_id,
                          'name': _name}

        elif self.args.drivable_only:
            _tmp_drivable = cv2.imread(drivable_label_path, cv2.IMREAD_GRAYSCALE)
            _tmp_drivable = self.convert_label(_tmp_drivable, self.drivable_mapping)
            _target_drivable = Image.fromarray(_tmp_drivable)

            sample = {   'image': _img, 
                'drivable_label': _target_drivable,
                          'name': _name}

        elif self.args.lane_only:
            _tmp_lane = cv2.imread(lane_label_path, cv2.IMREAD_GRAYSCALE)
            _tmp_lane = self.convert_label(_tmp_lane, self.lane_mapping)
            _target_lane = Image.fromarray(_tmp_lane)

            sample = {   'image': _img, 
                    'lane_label': _target_lane,
                          'name': _name}

        elif self.args.scenes_only:
            sample = {   'image': _img, 
                        'roadid': _scenes_id,
                          'name': _name}
        
        else:
            print('检查你的训练设置！')

        if self.split == 'train':
            return self.transform_tr(sample), _scenes_id
        elif self.split == 'val':
            return self.transform_val(sample), _scenes_id
        elif self.split == 'test':
            return self.transform_ts(sample)

    def convert_label(self, label, seg_map, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in seg_map.items():
                label[temp == k] = v
        else:
            for k, v in seg_map.items():
                label[temp == k] = v
        return label

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Multi_scale_aug(),
            tr.Rescale(ratio=self.args.ratio),
            # tr.Flip(),
            tr.RandomHorizontalFlip(), 
            tr.RandomGaussianBlur(),
            tr.Normalize(mean, std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Multi_scale_aug(),
            tr.Rescale(ratio=self.args.ratio),
            # tr.Flip(),
            tr.RandomHorizontalFlip(), 
            tr.RandomGaussianBlur(),
            tr.Normalize(mean, std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
#            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean, std),
            tr.ToTensor()])

        return composed_transforms(sample)

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

    bdd100k_train = BDD100kSegmentation(args, split='train')
    bdd100k_val = BDD100kSegmentation(args, split='val')
    dataloader = DataLoader(bdd100k_train, batch_size=2, shuffle=True, num_workers=12)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            drivable_gt = sample['drivable_label'].numpy()
            lane_gt = sample['lane_label'].numpy()
            segmap = np.array(drivable_gt[jj]).astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

