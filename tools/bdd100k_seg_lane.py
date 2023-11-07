from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import os,cv2
import json 
import numpy as np

from tqdm import tqdm


def poly2patch(poly2d, closed=False, alpha=1., color=None):
    moves = {'L': Path.LINETO,'C': Path.CURVE4}
    points = [p[:2] for p in poly2d]
    codes = [moves[p[2]] for p in poly2d]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else 'none',
        edgecolor=color,  # if not closed else 'none',
        lw=1 if closed else 2 * 1, alpha=alpha,
        antialiased=False, snap=True)

def get_areas_v0(objects):
    # print(objects['category'])
    return [o for o in objects
            if 'poly2d' in o and o['category'].startswith('area')]


def draw_drivable(objects, ax):
    plt.draw()

    objects = get_areas_v0(objects)
    for obj in objects:
        if obj['category'] == 'area/drivable':
            color = (1, 1, 1)
        # elif obj['category'] == 'area/alternative':
        #     color = (0, 1, 0)
        else:
            if obj['category'] != 'area/alternative':
                print(obj['category'])
            color = (0, 0, 0)
        # alpha = 0.5
        alpha = 1.0
        poly2d = obj['poly2d']
        ax.add_patch(poly2patch(poly2d, closed=True, alpha=alpha, color=color))

    ax.axis('off')


def filter_pic(data):
    for obj in data:
        if obj['category'].startswith('area'):
            return True
        else:
            pass
    return False

colors = [(0,0,255),(112,128,144),(0.255,255),(0,255,127),(255,255,0),(255,20,147),(255,0,255),(75,0,130),
          (0,0,255),(218,165,32),(245,222,179),(255,165,0),(255,69,0),(205,92,92)]

def get_slope(x1,y1,x2,y2):
    return (y2-y1) /(x2-x1 +0.0001)

def draw_lane(data,img):
    lane_object = [obj for obj in data if 'poly2d' in obj and (obj['category'].startswith("lane") and obj['category'].endswith("curb") is False and obj['category'].endswith("crosswalk") is False)]
    index = 0
    for obj in lane_object:
        poly2d = obj["poly2d"]
        print('poly2d',poly2d)
        for i,x_y in enumerate(poly2d):
            x,y,STR = float(x_y[0]),float(x_y[1]),x_y[2]
            cv2.circle(img,(int(x),int(y)),1,colors[index],5)
            if i == 0:
                x1,y1 = int(x),int(y)
            else:
                x2,y2 = int(x),int(y)
                slope = get_slope(x1,y1,x2,y2)
                if abs(slope) > 0.1:
                    cv2.line(img,(x1,y1),(x2,y2),colors[index],3,4)
                    cv2.putText(img,STR,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1.0,colors[index],1)
                x1,y1 = int(x),int(y)
        index +=1  
        if index >13:
            index =13
    return img,lane_object
                

def remain_lane(data):
    for obj in data:
        if obj['category'].startswith("lane") and obj['category'].endswith("curb") is False and obj['category'].endswith("crosswalk") is False:
            return True
        else:
            pass
    return False    


def main(mode="train"):
    image_dir = "/workspace/xiaozhihao/BiSeNet/BDD100K/images/{}".format(mode)
    label_dir = "/workspace/xiaozhihao/BiSeNet/BDD100K/det_annotations/100k/{}".format(mode)
    out_dir = '/workspace/xiaozhihao/{}'.format(mode)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    label_list = os.listdir(label_dir)  
    # val_pd = pd.read_json(open(val_json))
    # print(val_pd.head())
    i =0
    for val_json in tqdm(label_list):
        if i >5:
            break
        i +=1
        val_json = os.path.join(label_dir, val_json)
        print(val_json)
        val_pd = json.load(open(val_json))
        data = val_pd['frames'][0]['objects']
        if remain_lane(data):
            img_name = val_pd['name']
            out_path = os.path.join(out_dir, img_name+'.png')
            img_path = os.path.join(image_dir, img_name+'.jpg')
            img = cv2.imread(img_path)
            img_result,lane_object = draw_lane(data,img)
            if len(lane_object)>0:
                cv2.imwrite(out_path,img_result)
            else:
                pass
        else:
            pass

def main_ori(mode="train"):
    image_dir = "/workspace/xiaozhihao/BiSeNet/BDD100K/images/{}".format(mode)
    val_dir = "/workspace/xiaozhihao/BiSeNet/BDD100K/det_annotations/100k/{}".format(mode)
    out_dir = '/workspace/xiaozhihao/{}'.format(mode)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    val_list = os.listdir(val_dir)  
    # val_pd = pd.read_json(open(val_json))
    # print(val_pd.head())
    i =0
    for val_json in tqdm(val_list):
        if i >50:
            break
        i +=1
        val_json = os.path.join(val_dir, val_json)
        val_pd = json.load(open(val_json))
        data = val_pd['frames'][0]['objects']
        img_name = val_pd['name']

        remain = filter_pic(data)
        # if remain:
        dpi = 80
        w = 16
        h = 9
        image_width = 1280
        image_height = 720
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        out_path = os.path.join(out_dir, img_name+'.png')
        ax.set_xlim(0, image_width - 1)
        ax.set_ylim(0, image_height - 1)
        ax.invert_yaxis()
        ax.add_patch(poly2patch(
            [[0, 0, 'L'], [0, image_height - 1, 'L'],
            [image_width - 1, image_height - 1, 'L'],
            [image_width - 1, 0, 'L']],
            closed=True, alpha=1., color=(0, 0, 0)))
        if remain:
            draw_drivable(data, ax)
        fig.savefig(out_path, dpi=dpi)
        plt.close()
    else:
        pass

if __name__ == '__main__':
    # 生成车道线
    main(mode='train')
    # 生成可行驶区域
    # main_ori(mode='val')