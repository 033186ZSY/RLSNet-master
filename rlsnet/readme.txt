model_v1: resnet50, BUSD, 车道线10类
model_v2: resnet50-psa, resa, BUSD, 场景识别分支加入了relu, 并进行了权重初始化, 车道线10类
model_v3: resnet50-psa, resa, BUSD, 无relu，无权重初始化，车道线10类
model_v4: resnet50-psa, resa, BUSD, 无relu，无权重初始化，车道线10类, 自注意力蒸馏
model_v5: resnet50-psa, BUSD, 无relu，无权重初始化，车道线10类, 特征融合, 数据增强



            mIoU_drivable           mIoU_lane         accuracy    toltle_epoch   learn rate
model_v1    0.652606866(23)       0.227999577（24）   0.707（22）     50             0.01
model_v2    0.600483687（36）      0.200350375（74）   0.603          50             0.01
model_v3    0.639741484（2）       0.205624241（5）    0.701（5）      50             0.01
model_v4    0.658416681（10）      0.237705743（44）   0.717（23）     50             0.01
model_v5    0.668231（8）          0.222077（18）      0.720          50             0.01