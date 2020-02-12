# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import math
import numpy as np
import argparse
import functools
import torchvision.transforms as transforms
import torch
import paddle
import paddle.fluid as fluid
from utils import *
import six
import random


# 实现linf约束 输入格式都是tensor 返回也是tensor [1,3,224,224]
def linf_img_tenosr(o, adv, epsilon=16.0 / 256):
    o_img = tensor2img(o)
    adv_img = tensor2img(adv)

    clip_max = np.clip(o_img * (1.0 + epsilon), 0, 255)
    clip_min = np.clip(o_img * (1.0 - epsilon), 0, 255)

    adv_img = np.clip(adv_img, clip_min, clip_max)

    adv_img = img2tensor(adv_img)

    return adv_img


"""
Explaining and Harnessing Adversarial Examples, I. Goodfellow et al., ICLR 2015
实现了FGSM 支持定向和非定向攻击的单步FGSM


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""


def FGSM(adv_program, eval_program, gradients, o, input_layer, output_layer, momentum=0.5, step_size=1.0 / 256, epsilon=16.0 / 256,
         isTarget=False, target_label=0, use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={input_layer.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)


    # 计算梯度
    g = exe.run(adv_program,
                fetch_list=[gradients],
                feed={input_layer.name: o, 'label': target_label}
                )
    g = g[0][0]



    # print(g)

    if isTarget:
        adv = o - np.sign(g) * step_size * momentum
    else:
        adv = o + np.sign(g) * step_size * momentum

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv

def G_FGSM(adv_program, eval_program, gradients, o, input_layer, output_layer, step_size=1.0 / 256, epsilon=16.0 / 256,
         pix_num = 3*224*224, isTarget=False, target_label=0, use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={input_layer.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)


    # 计算梯度
    g = exe.run(adv_program,
                fetch_list=[gradients],
                feed={input_layer.name: o, 'label': target_label}
                )
    g = np.array(g[0][0])

    p = np.zeros(shape=g.shape, dtype=np.float32)

    # print(g.shape)     [3,224,224]
    # 更改一半的像素点
    # pix_num = 3*224*224/2

    for pix in range(int(pix_num)):
        # 获取最大值的坐标
        id_max = np.argmax(np.abs(g))
        pos = np.unravel_index(id_max, g.shape)
        a, b, c = pos
        # 令p上该坐标等于原值
        p[a][b][c] = g[a][b][c]
        # g上该坐标为0， 再次寻找g的最大值
        g[a][b][c] = 0
        # 打印输出找到的坐标位置与坐标值
        # if pix % 2000 == 0:
        #     print("the step is {0}, {1} point has found!, it's values is {2}".format(pix, pos, p[a][b][c]))

    # print(g)

    if isTarget:
        adv = o - np.sign(p) * step_size
    else:
        adv = o + np.sign(p) * step_size

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv



"""
Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras, 
and A. Vladu, ICLR 2018
实现了PGD 支持定向和非定向攻击的PGD


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""


def PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, step_size=2.0 / 256, epsilon=16.0 / 256,
        iteration=20, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={input_layer.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()

    for _ in range(iteration):

        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label}
                    )
        g = g[0][0]

        if isTarget:
            adv = adv - np.sign(g) * step_size
        else:
            adv = adv + np.sign(g) * step_size

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def PGD1(adv_program, eval_program, gradients, o, input_layer, output_layer, step_size=1.0 / 256, epsilon=16.0 / 256,
        iteration=20, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=output_layer,
                     feed={input_layer.name: o})

    o_label = result[0][0]  # 两个label 也是一样的

    o_label = np.argsort(o_label)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()

    for _ in range(iteration):

        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label}
                    )
        print(".............")
        print(np.array(g).shape)

        g = g[0][0]

        if isTarget:
            adv = adv - np.sign(g) * step_size
        else:
            adv = adv + np.sign(g) * step_size

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def M_PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, momentum=0.5, step_size=1.0 / 256,
          epsilon=16.0 / 256, iteration=20, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=output_layer,
                     feed={input_layer.name: o})

    o_label = result[0][0]  # 两个label 也是一样的

    o_label = np.argsort(o_label)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()
    S = 0
    for _ in range(iteration):
        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label}
                    )
        g = np.array(g[0][0])

        #print(g.shape)     [3,224,224]

        S = S * momentum + (g / np.mean(np.abs(g))) * (1-momentum)

        if isTarget:
            adv = adv - np.sign(S) * step_size
        else:
            adv = adv + np.sign(S) * step_size

    print("the current max point difference is {}".format(np.round(np.max(np.abs(adv-o)) * 255)))

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def G_PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, momentum=0.5, step_size=1.0 / 256,
          epsilon=16.0 / 256, iteration=20, pix_num=3*224*224, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=output_layer,
                     feed={input_layer.name: o})

    o_label = result[0][0]  # 两个label 也是一样的

    o_label = np.argsort(o_label)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()
    print(adv.shape)
    S = np.zeros(shape=[3,224,224], dtype=np.float32)
    for _ in range(iteration):
        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label}
                    )
        g = np.array(g[0][0])

        p = np.zeros(shape=g.shape, dtype=np.float32)


        #print(g.shape)     [3,224,224]
        #更改一半的像素点
        # pix_num = 3*224*224/2

        for pix in range(int(pix_num)):
            #获取最大值的坐标
            id_max = np.argmax(np.abs(g))
            pos = np.unravel_index(id_max, g.shape)
            a, b, c = pos
            #令p上该坐标等于原值
            p[a][b][c] = g[a][b][c]
            #g上该坐标为0， 再次寻找g的最大值
            g[a][b][c] = 0
            #打印输出找到的坐标位置与坐标值
            if pix % 20000 == 0:
                print("the step is {0}, {1} point has found!, it's values is {2}".format(pix, pos, p[a][b][c]))



        #S = S * momentum + (g / np.mean(np.abs(g))) * (1-momentum)
        S = S * momentum + (p / np.mean(np.abs(p))) * (1 - momentum)

        if isTarget:
            adv = adv - np.sign(S) * step_size
        else:
            adv = adv + np.sign(S) * step_size




    print("the current max point difference is {}".format(np.round(np.max(np.abs(adv-o)) * 255)))

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv

def L_PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, b1=0.9, b2=0.999, step_size=1.0 / 256,
          epsilon=16.0 / 256, iteration=20, pix_num=3*224*224, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=output_layer,
                     feed={input_layer.name: o})

    o_label = result[0][0]  # 两个label 也是一样的

    o_label = np.argsort(o_label)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()
    # print(adv.shape)

    M = 0
    V = 0

    for _ in range(iteration):
        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label}
                    )
        g = np.array(g[0][0])

        p = np.zeros(shape=g.shape, dtype=np.float32)


        for pix in range(int(pix_num)):
            #获取最大值的坐标
            id_max = np.argmax(np.abs(g))
            pos = np.unravel_index(id_max, g.shape)
            a, b, c = pos
            #令p上该坐标等于原值
            p[a][b][c] = g[a][b][c]
            #g上该坐标为0， 再次寻找g的最大值
            g[a][b][c] = 0
            #打印输出找到的坐标位置与坐标值
            # if pix % 2000 == 0:
            #     print("the step is {0}, {1} point has found!, it's values is {2}".format(pix, pos, p[a][b][c]))

        M = b1 * M + (1 - b1) * p
        V = b2 * V + (1 - b2) * np.square(p)

        M_ = M / (1 - np.power(b1, _ + 1))
        V_ = V / (1 - np.power(b2, _ + 1))
        R = M_ / (np.sqrt(V_) + 10e-9)


        if isTarget:
            adv = adv - np.sign(R) * step_size
        else:
            adv = adv + np.sign(R) * step_size

        adv = adv.astype('float32')


    print("the current max point difference is {}".format(np.round(np.max(np.abs(adv-o)) * 255)))

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv

def Full_L_PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, b1=0.9, b2=0.999, step_size=1.0 / 256,
          epsilon=16.0 / 256, iteration=20, pix_num=3*224*224, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=output_layer,
                     feed={input_layer.name: o})

    o_label = result[0][0]  # 两个label 也是一样的

    o_label = np.argsort(o_label)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()
    # print(adv.shape)

    M = 0
    V = 0

    for _ in range(iteration):
        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label}
                    )
        g = np.array(g[0][0])

        # p = np.zeros(shape=g.shape, dtype=np.float32)


        # for pix in range(int(pix_num)):
        #     #获取最大值的坐标
        #     id_max = np.argmax(np.abs(g))
        #     pos = np.unravel_index(id_max, g.shape)
        #     a, b, c = pos
        #     #令p上该坐标等于原值
        #     p[a][b][c] = g[a][b][c]
        #     #g上该坐标为0， 再次寻找g的最大值
        #     g[a][b][c] = 0
            #打印输出找到的坐标位置与坐标值
            # if pix % 2000 == 0:
            #     print("the step is {0}, {1} point has found!, it's values is {2}".format(pix, pos, p[a][b][c]))

        M = b1 * M + (1 - b1) * g
        V = b2 * V + (1 - b2) * np.square(g)

        M_ = M / (1 - np.power(b1, _ + 1))
        V_ = V / (1 - np.power(b2, _ + 1))
        R = M_ / (np.sqrt(V_) + 10e-9)


        if isTarget:
            adv = adv - np.sign(R) * step_size
        else:
            adv = adv + np.sign(R) * step_size

        adv = adv.astype('float32')


    print("the current max point difference is {}".format(np.round(np.max(np.abs(adv-o)) * 255)))

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def Multi_M_PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, b1=0.9, b2=0.999, pix_num = 3*224*224,
                step_size=1.0 / 256, epsilon=16.0 / 256, iteration=20, isTarget=False, target_label=0, use_gpu=True):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    results = exe.run(eval_program,
                      fetch_list=output_layer,
                      feed={input_layer.name: o})

    o_labels = [result[0] for result in results]  # 输出用
    o_labels = [np.argsort(o_label)[::-1][:1][0] for o_label in o_labels]



    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()
    M = 0
    V = 0
    for i in range(iteration):
        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: adv, 'label': target_label, 'origin_img': o}
                    )
        g = np.array(g[0][0])
        p = np.zeros(shape=g.shape, dtype=float)

        # print(g.shape)     [3,224,224]

        for pix in range(int(pix_num)):
            id_max = np.argmax(np.abs(g))
            pos = np.unravel_index(id_max, g.shape)
            a, b, c = pos
            p[a][b][c] = g[a][b][c]
            g[a][b][c] = 0

        M = b1 * M + (1 - b1) * p
        V = b2 * V + (1 - b2) * np.square(p)

        M_ = M / (1 - np.power(b1, i + 1))
        V_ = V / (1 - np.power(b2, i + 1))
        R = M_ / (np.sqrt(V_) + 10e-9)

        adv = adv + np.sign(R) * step_size
        adv = adv.astype('float32')

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv


def random_transform(adv):
    adv = adv.reshape(3, 224, 224)
    adv = torch.Tensor(adv)
    # adv = transforms.ToTensor()(adv)
    new_size = random.randint(190, 224)
    adv_img = transforms.ToPILImage()(adv)
    resized = transforms.Resize((new_size, new_size))(adv_img)
    adv = transforms.ToTensor()(resized)
    #print(adv.shape)
    diff = 224 - new_size
    for i in range(diff):
        w = random.random()
        h = random.random()
        pad = torch.nn.ZeroPad2d(padding=(round(w), round(1 - w), round(h), round(1 - h)))
        adv = pad(adv)
        #print(adv.shape)
    adv = adv.reshape(1, 3, 224, 224)
    return adv.numpy()


def T(adv, t):
    r = random.uniform(0, 1.0)
    if (r < t):
        #print("已做变换处理")
        return random_transform(adv)
    else:
        #print("未作变换处理")
        return adv


def T_PGD(adv_program, eval_program, gradients, o, input_layer, output_layer, t=0.6, b1=0.9, b2=0.999,
          step_size=1.0 / 256, epsilon=16.0 / 256, iteration=10, pix_num = 3*224*224/30, isTarget=False, target_label=0, use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={input_layer.name: o})
    result = result[0][0]
    #print("result:{}".format(result))
    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)

    adv = o.copy()
    M = 0
    V = 0
    for i in range(iteration):
        # 计算梯度
        g = exe.run(adv_program,
                    fetch_list=[gradients],
                    feed={input_layer.name: T(adv, t), 'label': target_label}
                    )
        g = np.array(g[0][0])
        p = np.zeros(shape=g.shape, dtype=float)

        #print("开始查点")
        for pix in range(int(pix_num)):
            id_max = np.argmax(np.abs(g))
            pos = np.unravel_index(id_max, g.shape)
            a, b, c = pos
            p[a][b][c] = g[a][b][c]
            g[a][b][c] = 0
        #print("查点结束")

        M = b1 * M + (1 - b1) * p
        V = b2 * V + (1 - b2) * np.square(p)

        M_ = M / (1 - np.power(b1, i + 1))
        V_ = V / (1 - np.power(b2, i + 1))
        R = M_ / (np.sqrt(V_) + 10e-9)

        adv = adv + np.sign(R) * step_size
        adv = adv.astype('float32')

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv

def T_FGSM(adv_program, eval_program, gradients, o, input_layer, output_layer, step_size=1.0 / 256, epsilon=16.0 / 256,
         pix_num = 3*224*224, t = 0.4, b1 = 0.9, b2 = 0.999, isTarget=False, target_label=0, use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={input_layer.name: o})
    result = result[0][0]

    o_label = np.argsort(result)[::-1][:1][0]

    if not isTarget:
        # 无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label = o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label, o_label))

    target_label = np.array([target_label]).astype('int64')
    target_label = np.expand_dims(target_label, axis=0)


    # 计算梯度
    g = exe.run(adv_program,
                fetch_list=[gradients],
                #feed={input_layer.name: o, 'label': target_label}
                feed = {input_layer.name: T(o, t), 'label': target_label}
                )
    g = np.array(g[0][0])

    p = np.zeros(shape=g.shape, dtype=np.float32)

    M = 0
    V = 0

    for pix in range(int(pix_num)):
        # 获取最大值的坐标
        id_max = np.argmax(np.abs(g))
        pos = np.unravel_index(id_max, g.shape)
        a, b, c = pos
        # 令p上该坐标等于原值
        p[a][b][c] = g[a][b][c]
        # g上该坐标为0， 再次寻找g的最大值
        g[a][b][c] = 0

    M = b1 * M + (1 - b1) * p
    V = b2 * V + (1 - b2) * np.square(p)

    M_ = M / (1 - np.power(b1, 1))
    V_ = V / (1 - np.power(b2, 1))
    R = M_ / (np.sqrt(V_) + 10e-9)


    if isTarget:
        adv = o - np.sign(R) * step_size
    else:
        adv = o + np.sign(R) * step_size

    # 实施linf约束
    adv = linf_img_tenosr(o, adv, epsilon)

    return adv