"""
此代码用于将BDD100K中的车道线处理成语义标签, 具体为：
只处理parallel的车道线, 利用两条车道线的中线代表该车道线的位置(8像素宽度)
车道线用三次贝塞尔曲线拟合
"""
import cv2
import os
import json
import numpy as np
from itertools import combinations
from tqdm import tqdm

label_mapping = {0: 255,
                                1: 20,  #背景这一类不需要预测
                                2: 100, 
                                3: 200,}

def observe_img_data(path):
    img_path = os.listdir('/workspace/xiaozhihao/BiSeNet/BDD100K/road_seg/train/')
    for data in img_path:
        img = cv2.imread('/workspace/xiaozhihao/BiSeNet/BDD100K/road_seg/train/'+data)
        h, w,_ = img.shape
        pixs = []
        for i in range(h):
            for j in range(w):
                pix = list(img[i][j])
                if pix not in pixs:
                    pixs.append(pix)
        print(pixs)

def observe_json_data(json_data_path):
    """观察json当中的数据分布"""
    train_set = os.listdir(os.path.join(json_data_path,'train'))
    # val_set = os.listdir(os.path.join(json_data,'val'))
    # datas = {'train_set':train_set, 'val_set':val_set}
    lane_type = []
    lane_style = []
    lane_direction = []
    txt = open('./fenxi.txt', 'w')
    for json_name in tqdm(train_set):
        print('开始分析{}的数据'.format(json_name))
        with open(os.path.join(json_data_path, 'train', json_name), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            object_data = json_data["frames"][0]["objects"]
            for data in object_data:
                if data["category"].startswith("lane") and data["category"] not in lane_type:
                    lane_type.append(data["category"])

                if data["category"].startswith("lane") and data["attributes"]["style"] not in lane_style:
                    lane_style.append(data["attributes"]["style"])

                if data["category"].startswith("lane") and data["attributes"]["direction"] not in lane_direction:
                    lane_direction.append(data["attributes"]["direction"])   

                if data["category"]=='lane/road curb':
                    print(data["attributes"]["direction"])
                    print(data["attributes"]["style"])
           
                    # if data["attributes"]["direction"] == "parallel":
                    #     print(len(data["poly2d"]))
                    #     txt.write(str(len(data["poly2d"]))+', '+str(data["poly2d"])+', '+json_name+', '+'\n')
                    
        fp.close()
    txt.write(str(lane_type)+'\n')
    txt.write(str(lane_style)+'\n')
    txt.write(str(lane_direction)+'\n')

def one_bezier_curve(a, b, t):
    """一阶贝塞尔曲线"""
    return (1 - t) * a + t * b

def n_bezier_curve(xs, n, k, t):
    """n阶贝塞尔曲线"""
    if n == 1:
        return one_bezier_curve(xs[k], xs[k + 1], t)
    else:
        return (1 - t) * n_bezier_curve(xs, n - 1, k, t) + t * n_bezier_curve(xs, n - 1, k + 1, t)

def bezier_curve(xs, ys, num, b_xs, b_ys, n):
    """在图像上绘制贝塞尔曲线"""
    t_step = 1.0 / (num - 1)
    # print(t_step)
    t = np.arange(0.0, 1 + t_step, t_step)
    # print(len(t))
    for each in t:
        b_xs.append(n_bezier_curve(xs, n, 0, each))
        b_ys.append(n_bezier_curve(ys, n, 0, each))
    return b_xs, b_ys

def gen_bdd_lane_deal(json_data_path, colorline_data_path, save_path, mode, threshold):
    """利用json中的"poly2d"绘制出车道线的语义标签,进行双线合并,但不考虑贝塞尔曲线的点"""
    data_set = os.listdir(os.path.join(json_data_path, mode))
    txt = open('/workspace/xiaozhihao/BiSeNet/fenxi.txt', 'w')
    if not os.path.exists(os.path.join(save_path, mode)):
        os.mkdir(os.path.join(save_path, mode))

    for json_name in tqdm(data_set):
        img = np.ones((720, 1280, 3), np.uint8)*255
        with open(os.path.join(json_data_path, mode, json_name), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            object_data = json_data["frames"][0]["objects"]
            """
            1/category和type是否一致
            2/车道线点数是否一致
            3/首尾两个点的距离是否小于阈值
            4/求两两坐标值的中心点为新的车道线特征点
            """
            ### 1、筛选符合要求的车道线（五种类型，两种线型，一种方向）
            lane_object = [obj for obj in object_data if 'poly2d' in obj and (obj['category'] in category and obj["attributes"]["direction"]=="parallel")]
            for ps in range(len(lane_object)):
                lane_object[ps]['poly2d'].sort()
            ### 2、将所有出现的类型与长度的车道线的属性和索引信息存到一个列表当中，含有重复的
            lane_type_num = [obj['category']+'_'+obj["attributes"]["style"]+'_'+ str(len(obj["poly2d"])) for obj in lane_object]
            ### 3、将同类型与长度的车道线的属性和索引信息存到一个字典dict_address当中，不含重复的
            list_same = []
            for x in lane_type_num:
                address_index = [i for i in range(len(lane_type_num)) if lane_type_num[i] == x]
                list_same.append([x, address_index])
            dict_address = dict(list_same)
            ### 4、计算相同类型和属性的车道线是否为同一条车道线的边缘线，如果是，则计算中心线+删除原来的两条边界线
            for key in dict_address:
                indexs = dict_address[key]
                if len(indexs)>=2: # 单根线就不用判断了
                    # print(indexs)
                    for p in combinations(indexs, 2): # 两两相互计算首尾两端的特征的距离
                        strat_0 = np.array(lane_object[p[0]]['poly2d'][0][:2]) # 第一条车道线的起点
                        strat_1 = np.array(lane_object[p[1]]['poly2d'][0][:2]) # 第二条车道线的起点
                        op1 = np.sqrt(np.sum(np.square(strat_0-strat_1)))      # 两条车道线起点的距离

                        end_0 = np.array(lane_object[p[0]]['poly2d'][-1][:2])  # 第一条车道线的终点
                        end_1 = np.array(lane_object[p[1]]['poly2d'][-1][:2])  # 第二条车道线的终点
                        op2 = np.sqrt(np.sum(np.square(end_0-end_1)))          # 两条车道线终点的距离
                        # print(op1, op2)
                        txt.write(str(op1)+', '+str(op2)+'\n')
                        if op1 < threshold and op2 < threshold:
                            lane1  = np.array([(float(i[0]), float(i[1])) for i in lane_object[p[0]]['poly2d']])
                            lane2  = np.array([(float(i[0]), float(i[1])) for i in lane_object[p[1]]['poly2d']])
                            center_lane = (lane1+lane2)/2
                            # print(lane_object)
                            lane_object[p[1]]['poly2d'] = center_lane.tolist()
                            lane_object[p[0]]['poly2d'] = center_lane.tolist()
                            # del lane_object[p[0]]
            # print(lane_object)
            ### 开始画线
            for data in lane_object:
                points = data["poly2d"]
                points_num = len(data["poly2d"])
                color_index = data["category"]+'_'+data["attributes"]["style"]
                num_category[color_index] += 1 
                if points_num == 2:
                    img = cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color_list[color_index], thickness=line_width[mode])
                elif points_num > 2:
                    xs, ys, b_xs, b_ys = [], [], [], []
                    for i in range(points_num):
                        xs.append(points[i][0])
                        ys.append(points[i][1]) 
                    n = len(ys)-1
                    b_xs, b_ys = bezier_curve(xs, ys, num, b_xs, b_ys, n)
                    for i in range(len(b_xs)):
                        img = cv2.circle(img, (int(b_xs[i]), int(b_ys[i])), 2, color_list[color_index], int(line_width[mode]/2))
        cv2.imwrite(os.path.join(save_path, mode, json_name.replace('.json','.png')), img)
        # cv2.imwrite(json_name.replace('.json','.png'), img)
    print(num_category)

def gen_bdd_lane_row(json_data_path, colorline_data_path, save_path, mode):
    """利用json中的"poly2d"绘制出车道线, 不进行双线合并"""
    data_set = os.listdir(os.path.join(json_data_path, mode))
    if not os.path.exists(os.path.join(save_path, mode)):
        os.mkdir(os.path.join(save_path, mode))
    color = (1,1,1)
    for json_name in data_set:
        print('开始处理{}的数据'.format(json_name))

        # img = cv2.imread(os.path.join(colorline_data_path, mode, json_name.split('.')[0]+'.png'))
        img = np.zeros((720,1280,3), np.uint8)

        with open(os.path.join(json_data_path, mode, json_name), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            object_data = json_data["frames"][0]["objects"]
            for data in object_data: # 读每条车道线的点
                if data["category"] in category and data["attributes"]["direction"] == "parallel":
                    points = data["poly2d"]
                    points_num = len(data["poly2d"])
                    if points_num == 2:
                        img = cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color, thickness=line_width[mode])
                    elif points_num > 2:
                        xs, ys, b_xs, b_ys = [], [], [], []
                        for i in range(points_num):
                            xs.append(points[i][0])
                            ys.append(points[i][1]) 
                        n = len(ys)-1
                        b_xs, b_ys = bezier_curve(xs, ys, num, b_xs, b_ys, n)
                        for i in range(len(b_xs)):
                            img = cv2.circle(img, (int(b_xs[i]), int(b_ys[i])), 2,color, line_width[mode]/2)
        cv2.imwrite(os.path.join(save_path, mode, json_name.replace('.json','.png')), img)
        fp.close()

def gen_bdd_lane_for_dataset(json_data_path, colorline_data_path, save_path, mode, color_list):
    """利用json中的"poly2d"绘制出车道线的语义标签, 不进行双线合并"""
    data_set = os.listdir(os.path.join(json_data_path, mode))
    if not os.path.exists(os.path.join(save_path, mode)):
        os.mkdir(os.path.join(save_path, mode))

    for json_name in tqdm(data_set):
        # print('开始处理{}的数据'.format(json_name))
        img = np.ones((720,1280,3), np.uint8)*255
        with open(os.path.join(json_data_path, mode, json_name), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            object_data = json_data["frames"][0]["objects"]
            for data in object_data: # 读每条车道线的点
                if data["category"] in category and data["attributes"]["direction"] == "parallel":
                    color_index = data["category"]+'_'+data["attributes"]["style"]
                    num_category[color_index] += 1 
                    points = data["poly2d"]
                    points_num = len(data["poly2d"])
                    if points_num == 2:
                        img = cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color_list[color_index], thickness=line_width[mode])
                    elif points_num > 2:
                        xs, ys, b_xs, b_ys = [], [], [], []
                        for i in range(points_num):
                            xs.append(points[i][0])
                            ys.append(points[i][1]) 
                        n = len(ys)-1
                        b_xs, b_ys = bezier_curve(xs, ys, num, b_xs, b_ys, n)
                        for i in range(len(b_xs)):
                            img = cv2.circle(img, (int(b_xs[i]), int(b_ys[i])), 2, color_list[color_index], line_width[mode]/2)
        cv2.imwrite(os.path.join(save_path, mode, json_name.replace('.json','.png')), img)
        fp.close()

if __name__ == '__main__':
    json_data_path = '/workspace/xiaozhihao/BiSeNet/BDD100K/det_annotations/100k/'
    img_data = '/workspace/xiaozhihao/BiSeNet/BDD100K/images'
    colorline_data_path = '/workspace/xiaozhihao/BiSeNet/BDD100K/det_annotations/100k/lane/colormaps'
    save_path = '/workspace/xiaozhihao/BiSeNet/BDD100K/xzh_bdd_lane/'
    num = 1000

    line_width = {'train':8,'val':2}

    category = ['lane/double yellow', 
                'lane/single white', 
                'lane/road curb', 
                # 'lane/crosswalk', 
                'lane/single yellow', 
                'lane/double white', 
                # 'lane/single other', 
                # 'lane/double other'
                ]

    color_list = {'lane/double yellow_solid' :(1, 1, 1),
                  'lane/double yellow_dashed':(2, 2, 2),
                  'lane/single white_solid'  :(3, 3, 3),
                  'lane/single white_dashed' :(4, 4, 4),
                  'lane/single yellow_solid' :(5, 5, 5),
                  'lane/single yellow_dashed':(6, 6, 6),
                  'lane/double white_solid'  :(7, 7, 7),
                  'lane/double white_dashed' :(8, 8, 8),
                  'lane/road curb_solid'     :(9, 9, 9),
                  'lane/road curb_dashed'    :(10, 10, 10)
                  }

    num_category = {'lane/double yellow_solid' :0,
                    'lane/double yellow_dashed':0,
                    'lane/single white_solid'  :0,
                    'lane/single white_dashed' :0,
                    'lane/single yellow_solid' :0,
                    'lane/single yellow_dashed':0,
                    'lane/double white_solid'  :0,
                    'lane/double white_dashed' :0,
                    'lane/road curb_solid'     :0,
                    'lane/road curb_dashed'    :0
                  }

    style = ['dashed','solid']

    if not os.path.exists(os.path.join(save_path)):
        os.mkdir(os.path.join(save_path))
    # observe_json_data(json_data_path)
    # gen_bdd_lane_row(json_data_path, img_data, save_path, mode = "val")
    gen_bdd_lane_deal(json_data_path, colorline_data_path, save_path, mode = "val", threshold = 100)
    # gen_bdd_lane_for_dataset(json_data_path, colorline_data_path, save_path, "train", color_list)
    # observe_img_data('/workspace/xiaozhihao/BiSeNet/BDD100K/road_seg/train/0a0c3694-f3444902_drivable_id.png')
    