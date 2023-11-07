import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

log_file_list = [
    '/workspace/xiaozhihao/RLSnet/checkpoint_of_XZH/(2022.12.17)Ablation/resnet50/resnet50_2022-12-21-09-48_train.log'
] #每次训练，都会产生一个log文件，如果断掉继续训练，则需要把新的log加进来

train_Loss = []
drivable_mIoU = []
lane_mIoU = []
accuracy = []

for log_file in log_file_list:
    for line in open(log_file,"r",encoding='UTF-8'):
        if line[24:29] == 'head:':
            line = line[29:]
            drivable_mIoU.append(float(line.split(', ')[3])*100)
            lane_mIoU.append(float(line.split(', ')[-4])*100)
            accuracy.append(float(line.split(', ')[-2])*100)
            train_Loss.append(float(line.split(', ')[-1]))

def draw_img():
    x = [i+1 for i in range(len(drivable_mIoU))]
    print(max(drivable_mIoU), ' epoch: {}'.format(drivable_mIoU.index(max(drivable_mIoU))))
    print(max(lane_mIoU), ' epoch: {}'.format(lane_mIoU.index(max(lane_mIoU))))
    print(max(accuracy), ' epoch: {}'.format(accuracy.index(max(accuracy))))

    fig,ax = plt.subplots(figsize=(12,9))
    ax.plot(x, train_Loss, color='black', linewidth=2, marker='o', label='train_loss')
    ax.set_xlabel('epoch', fontsize=20)
    ax.set_ylabel('loss', color='black', fontsize=20)


    plt.grid(True)
    ax2 = ax.twinx()
    ax2.plot(x, drivable_mIoU, color='red', linewidth=2, marker='^', label='drivable_miou')
    ax2.plot(x, lane_mIoU, color='green', linewidth=2, marker='s', label='lane_miou')
    ax2.plot(x, accuracy, color='blue', linewidth=2, marker='d', label='scenes_acc')
    ax2.set_ylabel('mIoU', color='black', fontsize=20)

    plt.title("RLSnet",fontsize=20)
    font_dict=dict(fontsize=16,
                  color='b',
                #   family='Times New Roman',
                  weight='light',
                #   style='italic',
                  )
    plt.text(drivable_mIoU.index(max(drivable_mIoU)), max(drivable_mIoU)+0.4, round(max(drivable_mIoU), 2), {'fontsize':16,'color':'r', 'weight':'bold'})
    plt.text(lane_mIoU.index(max(lane_mIoU)), max(lane_mIoU)+0.4, round(max(lane_mIoU), 2), {'fontsize':16,'color':'g','weight':'bold'})
    plt.text(accuracy.index(max(accuracy)), max(accuracy)+0.4, round(max(accuracy), 2), {'fontsize':16,'color':'b','weight':'bold'})

    plt.legend(prop={'size':16})
    plt.show()
    plt.savefig(log_file_list[0].replace('.log', '.png'))

def save_excel():
    dfData ={}
    dfData['epoch'] = [i+1 for i in range(len(drivable_mIoU))]
    dfData['drivable_mIoU'] = drivable_mIoU
    dfData['lane_mIoU'] = lane_mIoU
    dfData['accuracy'] = accuracy

    fileName = 'Baseline+SCM.xlsx'
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）
if __name__ == '__main__':
#    save_excel()
   draw_img()