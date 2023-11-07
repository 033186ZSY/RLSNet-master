import os
row_img_path = '/workspace/xiaozhihao/PIDNet-main/data/BDD100K/images'
label_path = '/xzh_bdd_lane/'

def pid_net():
    mode = {'train':10000 , 'val':1000} # 70000, 10000
    for key in mode:
        print(key)
        num = mode[key]
        i = 0
        txt = open("/workspace/xiaozhihao/PIDNet-main/data/BDD100K/"+key+".txt",'w')
        # row_img = os.listdir(row_img_path+'/'+key)
        # for fl in row_img:
        #     i +=1
        #     if i<=num:
        #         a = "/images/"+key+"/"+fl+' '+label_path+key+"/"+fl.split('.')[0]+'.png'+'\n'
        #         print(a)
        #         txt.write(a)
        # txt.close


        for line in open("/workspace/xiaozhihao/PIDNet-main/data/BDD100K/"+key+"_rlsnet.txt"):
            line = line.strip()
            a = "/images/"+key+"/"+line+'.jpg'+' '+label_path+key+"/"+line+'.png'
            # print(a)
            txt.write(a+'\n')
        txt.close

def rlsnet_txt():
    mode = {'train':10000 , 'val':3000} # 70000, 10000
    for key in mode:
        num = mode[key]
        i = 0
        txt = open("/workspace/xiaozhihao/RLSnet/BDD100K/"+key+'_'+"rlsnet.txt",'w')
        row_img = os.listdir(row_img_path+'/'+key)
        for fl in row_img:
            i +=1
            if i<=num:
                a = fl.split('.')[0]+'\n'
                print(a)
                txt.write(a)
        txt.close
        
# rlsnet_txt()
pid_net()