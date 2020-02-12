import cv2
from PIL import Image

list_file = r"D:\AI安全对抗\attack_example\input_image\val_list.txt"

list_pic = []

with open(list_file, "r") as f:
    for line in f:
        line = line.split("\t")
        list_pic.append(line[1].rstrip())


for img_path in list_pic:
    #print("output_image/" + img_path.split(".")[0] + ".png")
    img = cv2.imread("output_image/" + img_path.split(".")[0] + ".png")
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    matrix = cv2.getRotationMatrix2D((112, 112), 150, 1)
    adv_img = cv2.warpAffine(img, matrix, ((224,224)))
    print(adv_img.shape)
    cv2.imwrite("result/" + img_path.split(".")[0] + ".png", adv_img)
