# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
# mport deeplearning_backbone.paddlecv.model_provider as paddlecv
# 加载自定义文件
import models
from attack.attack_pp import FGSM, PGD, M_PGD, G_PGD, L_PGD, T_PGD
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments

with_gpu = os.getenv('WITH_GPU', '0') != '0'
#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)


add_arg('class_dim', int, 121, "Class number.")
add_arg('shape', str, "3,224,224", "output image shape")
add_arg('input', str, "./input_image/", "Input directory with images")
add_arg('output', str, "./final/v3/001/", "Output directory with images")

args = parser.parse_args()
print_arguments(args)

######Init args
global out  # 指定哪一个模型
out = None
image_shape = [int(m) for m in args.shape.split(",")]
class_dim = args.class_dim
input_dir = args.input
output_dir = args.output
model_name1 = "ResNeXt50_32x4d"

model_name2 = "MobileNetV2_x2_0"


model_params = "models_parameters/params"
val_list = 'val_list.txt'
use_gpu = False

#####################double_adv_program#######################
double_adv_program = fluid.Program()

#global Res_ratio  # Res 比重
Res_ratio = 0.8
with fluid.program_guard(double_adv_program):
    input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 设置为可以计算梯度
    input_layer.stop_gradient = False

    # model definition
    Res_model = models.__dict__[model_name1]()  # Res
    Res_out_logits = Res_model.net(input=input_layer, class_dim=class_dim)
    Res_out = fluid.layers.softmax(Res_out_logits)

    Inception_model = models.__dict__[model_name2]()  # Inception
    Inception_out_logits = Inception_model.net(input=input_layer, class_dim=class_dim)
    Inception_out = fluid.layers.softmax(Inception_out_logits)

    # place = fluid.CUDAPlace(0) if with_gpu else fluid.CPUPlace()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_params(executor=exe, dirname=model_params, main_program=double_adv_program)


init_prog(double_adv_program)

# 创建测试用评估模式
double_eval_program = double_adv_program.clone(for_test=True)

with fluid.program_guard(double_adv_program):
    label = fluid.layers.data(name="label", shape=[1], dtype='int64')
    Res_loss = fluid.layers.cross_entropy(input=Res_out, label=label)
    Inception_loss = fluid.layers.cross_entropy(input=Inception_out, label=label)
    loss = Res_loss * Res_ratio + (1 - Res_ratio) * Inception_loss
    gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]


######Inference
def inference(img, out):
    fetch_list = [o.name for o in out]

    result = exe.run(double_eval_program,
                     fetch_list=fetch_list,
                     feed={'image': img})
    # result = result[0]
    pred_label = [np.argmax(res[0]) for res in result]

    pred_score = []
    for i, pred in enumerate(pred_label):
        pred_score.append(result[i][0][pred].copy())
    return pred_label, pred_score


# untarget attack
def attack_nontarget_by_PGD(adv_prog, img, pred_label, src_label, out=None):
    # pred_label = [src_label, src_label]

    step = 8.0 / 256.0
    eps = 128.0 / 256.0
    while src_label in pred_label:

        # 生成对抗样本
        adv = T_PGD(adv_program=adv_prog, eval_program=double_eval_program, gradients=gradients, o=img,
                  input_layer=input_layer, output_layer=out, step_size=step, epsilon=eps, iteration=10,
                  t = 0.6, pix_num=224*224*3/30, isTarget=False, target_label=0, use_gpu=use_gpu)

        pred_label, pred_score = inference(adv, out)
        print("the current label is {}".format(pred_label))
        print("the current step is {}".format(step))
        #step += 1.0 / 256.0
        step *= 1.5
        if step > eps:
            break


    print("Test-score: {0}, class {1}".format(pred_score, pred_label))

    adv_img = tensor2img(adv)
    return adv_img


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
    mse = 0
    num = 1
    original_files = get_original_file(input_dir + val_list)

    f = open('log.txt', 'w')  # log

    for filename, label in original_files:

        img_path = input_dir + filename
        print("Image: {0} ".format(img_path))
        img = process_img(img_path)

        Res_result, Inception_result = exe.run(double_eval_program,
                                            fetch_list=[Res_out, Inception_out],
                                            feed={input_layer.name: img})
        Res_result = Res_result[0]
        Inception_result = Inception_result[0]

        r_o_label = np.argsort(Res_result)[::-1][:1][0]
        i_o_label = np.argsort(Inception_result)[::-1][:1][0]

        pred_label = [r_o_label, i_o_label]

        print("原始标签为{0}".format(label))
        print("Res result: %d, Inception result: %d" % (r_o_label, i_o_label))

        f.write("原始标签为{0}\n".format(label))
        f.write("Res result: %d, Inception result: %d\n" % (r_o_label, i_o_label))

        if r_o_label == int(label) and i_o_label == int(label):

            global Res_ratio

            Res_ratio = 0.8

            adv_img = attack_nontarget_by_PGD(double_adv_program, img, pred_label, label, out=[Res_out, Inception_out])

            image_name, image_ext = filename.split('.')
            ##Save adversarial image(.png)

            org_img = tensor2img(img)
            score = calc_mse(org_img, adv_img)

            #image_name += "MSE_{}".format(score)
            save_adv_image(adv_img, output_dir + image_name + '.png')
            mse += score

        elif r_o_label == int(label):  # Inception 预测错误
            print("filename:{}, Inception failed!".format(filename))

            Res_ratio = 1.0

            adv_img = attack_nontarget_by_PGD(double_adv_program, img, [r_o_label, 0], label, out=[Res_out])

            image_name, image_ext = filename.split('.')
            ##Save adversarial image(.png)

            org_img = tensor2img(img)
            score = calc_mse(org_img, adv_img)

            #image_name += "MSE_{}".format(score)
            save_adv_image(adv_img, output_dir + image_name + '.png')
            mse += score

        else:
            print("{0}个样本已为对抗样本, name为{1}".format(num, filename))
            score = 0
            f.write("{0}个样本已为对抗样本, name为{1}\n".format(num, filename))
            img = tensor2img(img)
            image_name, image_ext = filename.split('.')
            #image_name += "_un_adv_"
            save_adv_image(img, output_dir + image_name + '.png')
        print("this rext network weight is {0}".format(Res_ratio))
        num += 1
        print("the image's mse is {}".format(score))
        # break
    print("ADV {} files, AVG MSE: {} ".format(len(original_files), mse / len(original_files)))
    #print("ADV {} files, AVG MSE: {} ".format(len(original_files - num), mse / len(original_files - num)))
    f.write("ADV {} files, AVG MSE: {} ".format(len(original_files), mse / len(original_files)))
    f.close()


def main():
    gen_adv()


if __name__ == '__main__':
    main()
