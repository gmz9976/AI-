#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
import deeplearning_backbone.paddlecv.model_provider as paddlecv
#加载自定义文件
import models
from attack.attack_pp import FGSM, PGD
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments
with_gpu = os.getenv('WITH_GPU', '0') != '0'
#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim',        int,   121,                  "Class number.")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "./final/v3/030/",     "Input directory with images")
add_arg('output',           str,   "./final/v1/three/005full+model/",    "Output directory with images")

args = parser.parse_args()
print_arguments(args)


######Init args
image_shape = [int(m) for m in args.shape.split(",")]
class_dim=args.class_dim
input_dir = args.input
output_dir = args.output

model_name0 = "ResNeXt50_32x4d"
pretrained_model0 ="D:/python_adver/paddle1.5/ResNeXt50_32x4dxxxxxxxxxx"

model_name0 = "MobileNetV2_x2_0"
pretrained_model0 = "D:/python_adver/paddle1.6/MobileNetV2"

model_name0 = "InceptionV4"
pretrained_model0 = "D:/python_adver/paddle1.5/InceptionV4"

model_name0 ="VGG19"
pretrained_model0 ="D:/python_adver/paddle1.5/VGG19"

model_name0 ="DistResNet"
pretrained_model0 ="D:/python_adver/paddle1.5/DistResNet"

model_name0 ="SE_ResNeXt101_32x4d"
pretrained_model0 ="D:/python_adver/paddle1.5/SE_ResNeXt101_32x4d"

model_name0 ="ShuffleNetV2"
pretrained_model0 ="D:/python_adver/paddle1.5/ShuffleNetV2"
# model_name0 = "DarkNet53"
# pretrained_model0 = "D:/python_adver/DarkNet53"

# model_name0 = "DenseNet161"
# pretrained_model0 = "D:/python_adver/DenseNet161"
#
# model_name0= "DPN131"
# pretrained_model0 = "D:/python_adver/DPN131"

# model_name = "VGG16"
# pretrained_model = "D:/python_adver/VGG16"

# model_name8 = "ResNeXt101_32x8d_wsl"
# pretrained_model8 = "D:/python_adver/ResNeXt101_32x8d_wsl"


# model_name0 = "EfficientNetB0"
# pretrained_model0 = "D:/python_adver/EfficientNetB0"

# model_name0 = "SE_ResNet50_vd"
# pretrained_model0 = "D:/python_adver/SE_ResNet50_vd"


# model_name = "ShuffleNetV2_swish"
# pretrained_model = "D:/python_adver/ShuffleNetV2_swish"

# model_name22 = "ResNet50"
# pretrained_model22 = "D:/python_adver/ResNet50"

# model_name12 = "AlexNet"
# pretrained_model12 = "D:/python_adver/AlexNet"

# model_name444 = "SqueezeNet1_1"
# pretrained_model44 = "D:/python_adver/SqueezeNet1_1"


# model_name44 = "ResNet50_vd"
# pretrained_model44 = "D:/python_adver/ResNet50_vd"

# model_name555 ="DenseNet121"
# pretrained_model555 ="D:/python_adver/DenseNet121"

# model_name555 ="Xception65"
# pretrained_model555 ="D:/python_adver/Xception65"

# model_name6 ="EfficientNetB4"
# pretrained_model6 ="D:/python_adver/EfficientNetB4"

# model_name0 ="Res2Net50_26w_4s"
# pretrained_model0 ="D:/python_adver/Res2Net50_26w_4s"

# model_name0 ="HRNet_W32_C"
# pretrained_model0 ="D:/python_adver/HRNet_W32_C"

# model_name0 ="ResNeXt101_vd_32x4d"
# pretrained_model0 ="D:/python_adver/ResNeXt101_vd_32x4d"

# model_name0 ="ResNeXt152_vd_32x4d"
# pretrained_model0 ="D:/python_adver/ResNeXt152_vd_32x4d"
#
# model_name44 ="ResNeXt50_vd_64x4d"
# pretrained_model44 ="D:/python_adver/ResNeXt50_vd_64x4d"
#
# model_name77 = "ResNeXt50_vd_32x4d"
# pretrained_model77 = "D:/python_adver/ResNeXt50_vd_32x4d"

# model_name0 ="ShuffleNetV2_x2_0"
# pretrained_model0 ="D:/python_adver/ShuffleNetV2_x2_0"

# model_name0 ="VGG19"
# pretrained_model0 ="D:/python_adver/VGG19"

# model_name1 ="SENet154_vd"
# pretrained_model1 ="D:/python_adver/SENet154_vd"

# model_name ="GoogleNet"
# pretrained_model ="D:/python_adver/GoogleNet"

model_name0 ="ResNeXt152_64x4d"
pretrained_model0 ="D:/python_adver/paddle1.6/ResNeXt152_64x4d"

model_name0 ="ResNeXt101_32x32d_wsl"
pretrained_model0 ="D:/python_adver/paddle1.6/ResNeXt101_32x32d_wsl"

model_name = "DARTS_4M"
pretrained_model = "D:/python_adver/paddle1.6/DARTS_4M"

model_name0 = "DARTS_6M"
pretrained_model0 = "D:/python_adver/paddle1.6/DARTS_6M"

val_list = 'val_list.txt'
use_gpu=False

######Attack graph

adv_program=fluid.Program()
#完成初始化
with fluid.program_guard(adv_program):

    input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 设置为可以计算梯度
    input_layer.stop_gradient = False

    #model definition
    model = models.__dict__[model_name]()
    #model = paddlecv.get_model("inceptionv4")

    out_logits = model.net(input=input_layer, class_dim=class_dim)

    out = fluid.layers.softmax(out_logits)

    # place = fluid.CUDAPlace(0) if with_gpu else fluid.CPUPlace()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_params(executor=exe, dirname=pretrained_model, main_program=adv_program)


#设置adv_program的BN层状态
init_prog(adv_program)

#创建测试用评估模式
eval_program = adv_program.clone(for_test=True)

### 定义梯度
with fluid.program_guard(adv_program):
    label = fluid.layers.data(name="label", shape=[1] ,dtype='int64')
    loss = fluid.layers.cross_entropy(input=out, label=label)
    gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]



######Inference
def inference(img):
    fetch_list = [out.name]

    result = exe.run(eval_program,
                     fetch_list=fetch_list,
                     feed={ 'image':img })
    result = result[0][0]
    pred_label = np.argmax(result)
    pred_score = result[pred_label].copy()
    return pred_label, pred_score


####### Main #######
def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files




def gen_adv():
    original_files = get_original_file(input_dir + val_list)
    test_acc = 0

    print("the model's name is {}".format(model_name))

    for filename, label in original_files:
        img_path = input_dir + filename
        print("Image: {0} ".format(img_path))
        img=process_img(img_path)

        result = exe.run(eval_program,
                         fetch_list=[out],
                         feed={input_layer.name: img})
        result = result[0][0]

        o_label = np.argsort(result)[::-1][:1][0]

        print("原始标签为{0}, {1}网络模型下标签为{2}".format(label, model_name, o_label))

        if o_label == int(label):
            test_acc += 1


    acc = test_acc / 120.0
    print("the acc num is {0}".format(test_acc))
    print("the model name is {0}, the acc is {1}".format(model_name, acc))




def main():
    gen_adv()


if __name__ == '__main__':
    main()
