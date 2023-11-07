from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

_C.cudnn = CN()
_C.cudnn.benchmark = True
_C.cudnn.deterministic = False 
_C.cudnn.enabled = True

_C.dataset = CN()
_C.dataset.root = '/workspace/xiaozhihao/BiSeNet/BDD100K'
_C.dataset.pretrained_model_path = None
_C.dataset.save_model_path = '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/'

_C.train = CN()
_C.train.num_epochs = 50
_C.train.epoch_start_i = 0
_C.train.checkpoint_step = 1
_C.train.validation_step = 1
_C.train.learning_rate = 0.01
_C.train.crop_height = 720
_C.train.crop_width = 1280
_C.train.ratio = 1
_C.train.num_workers = 4
_C.train.batch_size = 1
_C.train.use_gpu = True
_C.train.cuda = 0
_C.train.note = ''
_C.train.flip = True
_C.train.multi_scale: True
_C.train.scale_factor: 16

_C.loss = CN()
_C.loss.optimizer = 'sgd'
_C.loss.ignore_label = 255
_C.loss.drivable_loss = 'crossentropy'
_C.loss.lane_loss = 'ohem'
_C.loss.scenes_loss = 'crossentropy'

_C.model = CN()
_C.model.version = 'model_v5'
_C.model.backbone = '50'
_C.model.use_psa = True
_C.model.num_workers = 12
_C.model.road_classes = 3
_C.model.lane_classes = 10
_C.model.scenes_classes = 4

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)