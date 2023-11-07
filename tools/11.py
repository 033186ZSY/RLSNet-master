import cv2
import os
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
vout = cv2.VideoWriter('demo.avi', fourcc , 8, (1280, 720))
img_files = os.listdir("/workspace/xiaozhihao/RLSnet/demo_vedio")
for img in img_files:
    if img.endswith('.jpg'):
        vis = cv2.imread("/workspace/xiaozhihao/RLSnet/demo_vedio/" + img)
        vout.write(vis)

vout.release()