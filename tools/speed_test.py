import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from rlsnet.model_ablation import RLSNet
import argparse
import os

def main(params):
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--use_psa', type=bool, default=False, help='path to save model')
    parser.add_argument('--drivable_only', action='store_true', default=False, help='path to save model')
    parser.add_argument('--lane_only', action='store_true', default=False, help='path to save model')
    parser.add_argument('--scenes_only', action='store_true', default=False, help='path to save model')
    parser.add_argument('--whole', action='store_true', default=False, help='path to save model')

    parser.add_argument('--use_BFA', type=bool, default=True, help='是否使用BFA')
    parser.add_argument('--use_FAM', type=bool, default=True, help='是否使用FAM')
    parser.add_argument('--use_SCM', type=bool, default=True, help='是否使用SCM')
    parser.add_argument('--use_BUSD', type=bool, default=True, help='是否使用BUSD')

    args = parser.parse_args(params)

    torch.cuda.set_device(0)
    model = RLSNet(args)
    model.cuda()
    model.eval()

    iterations = None
    input = torch.randn(1, 3, 384, 640).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)

if __name__ == '__main__':
    params = [
        '--num_epochs', '60',
        '--learning_rate', '0.01',
        '--data', '/workspace/xiaozhihao/RLSnet/BDD100K',
        '--num_workers', '4',
        '--road_classes', '3',
        '--lane_classes', '10',
        '--scenes_classes', '4',
        '--cuda', '1', 
        '--batch_size', '18',  # 6 for resnet101, 20 for resnet18
        '--save_model_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.27)model_v7/',
        '--backbone', '50',  # only support resnet50 and resnet101
        '--note', '384*640_bs-18_34',
        # '--drivable_only',   # 只训练可行驶区域分支
        # '--lane_only', 
        # '--scenes_only',
        '--whole',
    ]
    main(params)

