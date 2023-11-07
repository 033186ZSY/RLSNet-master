"""修改时间2022.10.10"""
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

drivable_color_map = {'Not drivable':               [0, 0, 0], 
              'Drivable area':              [0, 255, 0], 
              'Alternative drivable area': [255, 0, 0]}

lane_color_map = {'background' :              (0, 0, 0),
                'lane/double yellow_solid' :  (0, 255, 255),
                  'lane/double yellow_dashed':(0,44,83),
                  'lane/single white_solid'  :(255, 255, 255),
                  'lane/single white_dashed' :(255,165,16),
                  'lane/single yellow_solid' :(65,183,172),
                  'lane/single yellow_dashed':(12,132,198),
                  'lane/double white_solid'  :(0, 0, 255),
                  'lane/double white_dashed' :(255,189,102),
                  'lane/road curb_solid'     :(247,77,77),
                  'lane/road curb_dashed'    :(36,85,164)}

road_ca = ['residential', 'highway', 'city street', 'other']

def predict_on_image(model, args, image):
    image = cv2.resize(image, (args.crop_width, args.crop_height))
    # resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize_det = resize.to_deterministic()
    # image = resize_det.augment_image(image)
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

def tuli(lane_predict):
    pts = np.array([(940,10),(1240,10),(1240,230),(940,230)],np.int32)
    img = cv2.fillPoly(lane_predict, [pts],(255,255,255))

    i = 1
    for key in lane_color_map:
        if key != 'background':
            i += 2
            img = cv2.line(img, (960,10*i), (1010, 10*i), lane_color_map[key], thickness=10)
            img = cv2.putText(img, key.split('/')[-1], (1020, 10*i+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    return img

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
    # fl = os.listdir('/workspace/xiaozhihao/RLSnet/demo_vedio/000e0252-8523a4a9')
    # for name in fl:
    #     raw = cv2.imread('/workspace/xiaozhihao/RLSnet/demo_vedio/000e0252-8523a4a9/'+name, -1)
        
    #     drivable_predict, lane_predict, scenes_id = predict_on_image(model, args, raw)
    #     road_scen = road_ca[scenes_id]
        
    #     result = cv2.addWeighted(drivable_predict, 0.5, lane_predict, 1, 3)
    #     result = cv2.addWeighted(result, 0.7, raw, 0.7, 3)
    #     result = tuli(result)
    #     result = cv2.circle(result, (80, 91), 4, (255,255,0), 12)
    #     result = cv2.putText(result, road_scen, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    #     cv2.imwrite(args.save_path + name, result)  


    for key in datas:
        for name in datas[key]:
            raw = cv2.imread(os.path.join(args.data, key, name + '.jpg'), -1)
            
            drivable_predict, lane_predict, scenes_id = predict_on_image(model, args, raw)
            road_scen = road_ca[scenes_id]
            
            result = cv2.addWeighted(drivable_predict, 0.5, lane_predict, 1, 3)
            result = cv2.addWeighted(result, 0.7, raw, 0.7, 3)
            result = tuli(result)
            result = cv2.circle(result, (80, 91), 4, (255,255,0), 12)
            result = cv2.putText(result, road_scen, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.imwrite(args.save_path + args.note + '_' + key+'_'+ name + '.jpg', result)


if __name__ == '__main__':
    test_set_list = ['cafb3b52-f34e5454','cb02bc12-00b2fbbb','cb3e5d46-784f9106','cb5fa573-fa7df890', 'cb6e5ba7-91566c19',
                    'cb7bffcf-5e7c5677','cb32ec15-6e369f96','cb37bfff-1b87d685', 'cb38bda4-26729b52', 'cb55a21f-cea4eb6a', 
                    'cb58f48b-957bfd2c', 'cb61fa6c-04f20136', 'cb86b1d9-7735472c']

    train_set_list = [
    '0ace96c3-48481887','00f89335-2ef7949d', '00fc910e-bce87172', '0a022bf4-be711762', 
    '0a62ffee-85e97e08', '0a172b0e-2af0d158', '0a851459-1da95e63','00adbb3f-7757d4ea', 
    '0d921414-1f2de3de', '0d660646-e58f93af','0a851459-1da95e63', '0da41df0-2e70a8a9', 
    '0d538703-23c86b77', '0d538703-ccd64878','0a172b0e-2af0d158', '0a022bf4-be711762', 
    '0da41df0-26725c58', '0dd60257-50384fb0','0d606092-486670ee', '0ddfea57-b0fe6132', 
    '0d921414-02b0a7ad', '0dd61039-a1a18769','0a62ffee-85e97e08', '0e1d00ca-c5faefa0',
    '0ad67e35-8628df27', '0b3d47ed-6cb90089', '0a172b0e-2af0d158', '00abd8a7-ecd6fc56', 
    '00de601c-858a8a8d'
                    ]
    val_set_list = ['c8476b52-383b95f4', 'b519a9a4-6dd5b989']

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
        '--note', '_'
    ]
    
    main(params)