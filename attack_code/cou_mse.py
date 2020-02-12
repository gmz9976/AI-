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
from attack.attack_pp import FGSM, PGD, M_PGD
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments
with_gpu = os.getenv('WITH_GPU', '0') != '0'
#######parse parameters
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim',        int,   120,                  "Class number.")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "./input_image/",     "Input directory with images")
add_arg('output',           str,   "./final/v3/033/",    "Output directory with images")

args = parser.parse_args()
print_arguments(args)


######Init args
image_shape = [int(m) for m in args.shape.split(",")]
class_dim=args.class_dim
input_dir = args.input
output_dir = args.output


val_list = 'val_list.txt'
use_gpu=False



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
    original_files = get_original_file(input_dir + val_list)


    for filename, label in original_files:

        img_path1 = input_dir + filename
        img_path2 = output_dir + filename.split('.')[0] + '.png'
        print("Image: {0} ".format(img_path1))
        img1=process_img(img_path1)
        img2=process_img(img_path2)

        img1 = tensor2img(img1)
        img2 = tensor2img(img2)


        score = calc_mse(img1, img2)
        mse += score


    print("ADV {} files, AVG MSE: {} ".format(len(original_files), mse/len(original_files)))


def main():
    #gen_adv(0)
    gen_adv()


if __name__ == '__main__':
    main()
