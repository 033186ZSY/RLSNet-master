
import cv2
import torchvision.transforms as transforms
import util.custom_transforms as tr
from PIL import Image
from torch.utils.data import Dataset
import os
import json

mean =[0.485, 0.456, 0.406]
std  =[0.229, 0.224, 0.225]

class BaseDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.root = args.data
        self.split = split
        self.resize_shape = [384, 640]

        self.images_base = os.path.join(self.root, 'images', self.split)
        self.drivable_seg = os.path.join(self.root, 'road_seg', self.split)
        self.lane_seg = os.path.join(self.root, 'xzh_bdd_lane', self.split)
        self.json_base = os.path.join(self.root, 'det_annotations', '100k', self.split)

        self.class_names = ['Not drivable', 'Drivable area', 'Alternative drivable area'] 
        self.ignore_index = 255
        self.db = []
        self.road_map = ['residential', 'highway', 'city street', 'other']
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

    def convert_label(self, label, seg_map, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in seg_map.items():
                label[temp == k] = v
        else:
            for k, v in seg_map.items():
                label[temp == k] = v
        return label

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
            # tr.Rescale(ratio=self.args.ratio),
            # tr.Flip(),
            # tr.RandomHorizontalFlip(), 
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean, std),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __getitem__(self, index):
        all_data = self.db[index]
        _img = Image.open(all_data['image']).convert('RGB')
        _img = _img.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)
        with open(all_data['json_label']) as f:
            data = json.load(f)

        road = data['attributes']['scene']
        if road in self.road_map:
            _scenes_id = self.road_map.index(road)
        else:
            _scenes_id = 3

        if self.split == 'test':
            sample = {'image': _img, 'name': all_data['image']}

        elif self.args.whole:
            _tmp_drivable = cv2.imread(all_data['drivable_label'], cv2.IMREAD_GRAYSCALE)
            _tmp_drivable = self.convert_label(_tmp_drivable, self.drivable_mapping)
            _target_drivable = Image.fromarray(_tmp_drivable)
            _target_drivable = _target_drivable.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)

            _tmp_lane = cv2.imread(all_data['lane_label'], cv2.IMREAD_GRAYSCALE)
            _tmp_lane = self.convert_label(_tmp_lane, self.lane_mapping)
            _target_lane = Image.fromarray(_tmp_lane)
            _target_lane = _target_lane.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)

            sample = {   'image': _img, 
                'drivable_label': _target_drivable, 
                    'lane_label': _target_lane, 
                        'roadid': _scenes_id,
                          'name': all_data['image']}

        elif self.args.drivable_only:
            _tmp_drivable = cv2.imread(all_data['drivable_label'], cv2.IMREAD_GRAYSCALE)
            _tmp_drivable = self.convert_label(_tmp_drivable, self.drivable_mapping)
            _target_drivable = Image.fromarray(_tmp_drivable)
            _target_drivable = _target_drivable.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)

            sample = {   'image': _img, 
                'drivable_label': _target_drivable,
                          'name': all_data['image']}

        elif self.args.lane_only:
            _tmp_lane = cv2.imread(all_data['lane_label'], cv2.IMREAD_GRAYSCALE)
            _tmp_lane = self.convert_label(_tmp_lane, self.lane_mapping)
            _target_lane = Image.fromarray(_tmp_lane)
            _target_lane = _target_lane.resize((self.resize_shape[1], self.resize_shape[0]), resample=0)

            sample = {   'image': _img, 
                    'lane_label': _target_lane,
                          'name': all_data['image']}

        elif self.args.scenes_only:
            sample = {   'image': _img, 
                        'roadid': _scenes_id,
                          'name': all_data['image']}
        
        else:
            print('检查你的训练设置！')

        if self.split == 'train':
            return self.transform_tr(sample), _scenes_id
        elif self.split == 'val':
            return self.transform_val(sample), _scenes_id
        elif self.split == 'test':
            return self.transform_ts(sample)

    @staticmethod
    def collate_fn(batch):
        samples, _scenes_id = zip(*batch)
        return samples, _scenes_id