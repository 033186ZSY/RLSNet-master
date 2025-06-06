# RLSNet
RLSNet based on pytorch 0.4.1 and python 3.6

## Dataset  
Download BDD100K dataset from (https://opendatalab.com/OpenDataLab/BDD100K)) 

  
## Pretrained model  
 `best_dice_loss_miou_0.655.pth` in checkpoint_of_XZH 

## Demo  
```
python /workspace/xiaozhihao/RLSnetv3/tools/zsy_demo.py
```  

### Result  
Result in demo result
## Train
```
python /workspace/xiaozhihao/RLSnetv3/tools/train_rlsnet_v7.py
```  
Use **tensorboard** to see the real-time loss and accuracy  

## Test
```
python /workspace/xiaozhihao/RLSnetv3/my_test.py
```

