### 项目地址：https://aistudio.baidu.com/aistudio/projectdetail/247996

### WP地址：https://mzgao.blog.csdn.net/article/details/104066620


### 环境需求：
- Python3.7
- paddle1.5
- cv2
- numpy
- torchvision
- torch

#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch

#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision


#### 运行： python3.7 attack_code/attack_second.py

#### 本次运行为一次添加黑盒模型的效果。部分参数配置：step=8.0/256, eps=128.0/256, t=0.6,使用T-FGSM算法

为了最大程度上模拟评测系统中的黑盒模型，训练了以下模型

- ./paddle1.5/InceptionV4
- ./paddle1.5/VGG19
- ./paddle1.5/DistResNet
- ./paddle1.5/SE_ResNeXt101_32x4d
- ./paddle1.5/SERes32x4d
- ./paddle1.6/Alex
- ./paddle1.6/DarkNet53
- ./paddle1.6/EfficientNetB4
- ./paddle1.6/Densenet161
- ./paddle1.6/DPN131
- ./paddle1.6/ResNext101_32x8d_wsl
- ./paddle1.6/SE_ResNet50_vd
- ./paddle1.6/shuffleNetV2_swish
- ./paddle1.6/Res2Net50_26w_4s
- ./paddle1.6/HRNet_W32_C
- ./paddle1.6/ResNeXt101_vd_32x4d
- ./paddle1.6/SEnet154_vd
- ./paddle1.6/shuffleV2_x2_0
- ./paddle1.6/DenseNet264
- ./paddle1.6/HRNet_W64_C
- ./paddle1.6/DARTS_4M(已上传)
- ./paddle1.6/DARTS_6M（已上传）

#### 以模拟黑盒模型最大程度图片为基础，添加训练的灰盒模型，部分参数配置：step=1.5/256, eps=128/256, t=0, iteration=8,使用T-PGD算法 

为了最大程度模拟灰盒模型，训练以下模型

- ./paddle1.5/ResNeXt50_32x4dxx（已上传）
- ./paddle1.5/ResNeXt50_32x4dxxx（已上传）
- ./paddle1.5/ResNeXt50_32x4dxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxxxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxxxxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxxxxxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxxxxxxxxx
- ./paddle1.5/ResNeXt50_32x4dxxxxxxxxxxxxx

注：这些灰盒模型是经由白盒模型参数resnext50经过对抗训练得来，每次都是扩充样本后，重新训练生成的模型参数
