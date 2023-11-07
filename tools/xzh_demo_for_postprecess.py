"""修改时间2022.10.10"""
from ast import While
import cv2
import sys 
sys.path.append('./')
import argparse
from rlsnet.model_v7 import RLSNet
import os
import torch
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, colour_code_segmentation
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
                    11: 8, #可行驶区域
                    12: 9 #可选行驶区域
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


def predict_on_image(model, args, image):
    # pre-processing on image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
    # predict
    model.eval()
    out_drivable, out_lane, output_scenes = model(image.cuda())

    scenes_id = np.argmax(output_scenes.cpu().detach().numpy())

    drivable_predict = out_drivable.squeeze()
    drivable_predict = reverse_one_hot(drivable_predict).cpu()
    drivable_predict = colour_code_segmentation(np.array(drivable_predict), drivable_color_map)
    drivable_predict = cv2.resize(np.uint8(drivable_predict), (1280, 720))

    lane_predict = out_lane.squeeze()
    lane_predict = reverse_one_hot(lane_predict).cpu()
    lane_predict = colour_code_segmentation(np.array(lane_predict), lane_color_map)
    lane_predict = cv2.resize(np.uint8(lane_predict), (1280, 720))

    return drivable_predict, lane_predict, scenes_id


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
    
    # sum_predict = convert_label(sum_predict, lane_mapping, inverse=True)
    # img = Image.fromarray(sum_predict)
    # img.save('tesy.png')



def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='(2022.10.9)train_bdd_road', help='demo图片命名的备注')
    parser.add_argument('--dataset', type=str, default="BDD100K", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default = 384, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default = 640, help='Width of cropped/resized input image to network')
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
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--ignore_label', type=int, default=255, help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--use_psa', type=bool, default=True, help='path to save model')
    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # model = BiSeNet(args.num_classes, args.context_path, road=True)
    model = RLSNet(args)
    # print(model)
    model.cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')
    
    # predict on image
    for key in datas:
        for name in datas[key]:
            raw = cv2.imread(os.path.join(args.data, key, name + '.jpg'), -1)
            drivable_predict, lane_predict, scenes_id = predict_on_image(model, args, raw)
            road_scen = road_ca[scenes_id]
            # cv2.imwrite(args.save_path + 'drivable_predict' + '_' + key + '_'+ name + '.png', drivable_predict)
            # cv2.imwrite(args.save_path + 'lane_predict' + '_' + key + '_'+ name + '.png', lane_predict)
            
            drivable_predict = drivable_post_process(drivable_predict, lane_predict)
            drivable_predict = cv2.cvtColor(drivable_predict, cv2.COLOR_GRAY2BGR)
            
            for i in range(720):
                for j in range(1280):
                    if drivable_predict[i][j][0]  == 12:
                        drivable_predict[i][j][0] = 255

                    if drivable_predict[i][j][1] == 11:
                        drivable_predict[i][j][1] =255

            result = cv2.addWeighted(drivable_predict, 0.5, lane_predict, 1, 3)
            result = cv2.addWeighted(result, 0.7, raw, 0.7, 3)
            result = cv2.circle(result, (80, 91), 4, (255,255,0), 12)
            result = cv2.putText(result, road_scen, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.imwrite(args.save_path + args.note + '_' + key+'_'+ name + '.jpg', result)

def mask_image( img, mask ):
   newImg = img.copy()
   newImg[:,:] = img[:,:] * mask[:,:]
   return newImg
if __name__ == '__main__':
    test_set_list = ['cb55a21f-cea4eb6a'
        # 'cb37bfff-1b87d685', 'cb38bda4-26729b52','cb86b1d9-7735472c', 'cb02bc12-00b2fbbb'
        ]
    train_set_list = [    '0ace96c3-48481887','00f89335-2ef7949d', '00fc910e-bce87172', '0a022bf4-be711762', 
    '0a62ffee-85e97e08', '0a172b0e-2af0d158', '0a851459-1da95e63','00adbb3f-7757d4ea', 
    '0d921414-1f2de3de', '0d660646-e58f93af','0a851459-1da95e63', '0da41df0-2e70a8a9', 
    '0d538703-23c86b77', '0d538703-ccd64878','0a172b0e-2af0d158', '0a022bf4-be711762', 
    '0da41df0-26725c58', '0dd60257-50384fb0','0d606092-486670ee', '0ddfea57-b0fe6132', 
    '0d921414-02b0a7ad', '0dd61039-a1a18769','0a62ffee-85e97e08', '0e1d00ca-c5faefa0',
    '0ad67e35-8628df27', '0b3d47ed-6cb90089', '0a172b0e-2af0d158', '00abd8a7-ecd6fc56', 
    '00de601c-858a8a8d']

    val_set_list = []

    datas = {
        'train':train_set_list, 
        # 'test':test_set_list, 
        # 'val':val_set_list
        }
    
    params = [
        '--data', '/workspace/xiaozhihao/RLSnet/BDD100K/images/',
        '--num_workers', '12',
        '--road_classes', '3',
        '--lane_classes', '10',
        '--scenes_classes', '4',
        '--cuda', '0', 
        '--batch_size', '1',  # 6 for resnet101, 12 for resnet18
        '--checkpoint_path', '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.10.27)model_v7/384*640_bs-18-PSA/53.pth',
        '--save_path', '/workspace/xiaozhihao/RLSnet/demo_results/',
        '--backbone', '18',  # only support resnet50 and resnet101
        '--ignore_label', '255',
        '--note', '2022.11.10_post_deal'
    ]

    # drivable_predict = cv2.imread('/workspace/xiaozhihao/RLSnet/demo_results/drivable_predict_train_00abd8a7-ecd6fc56.png')
    # lane_predict = cv2.imread('/workspace/xiaozhihao/RLSnet/demo_results/lane_predict_train_00abd8a7-ecd6fc56.png')
    # lane_drivable_post_process(drivable_predict, lane_predict)
    main(params)