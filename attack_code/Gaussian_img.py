import cv2
import skimage.util as ski
from PIL import Image

input_dir = "./input_image/"
output_dir = "./final/gaussian_img/"
fix_name = 'gaussian'


def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files






def main():
    files = get_original_file(input_dir + 'val_list.txt')
    print(files)
    for file in files:
        filename, label = file
        print(filename)
        img = cv2.imread(input_dir + filename)
        img_gaussion_001 = ski.random_noise(img, mode="gaussian", seed=None, clip=True, mean=0, var=0.003)
        img_gaussion_002 = ski.random_noise(img, mode="gaussian", seed=None, clip=True, mean=0, var=0.005)
        img_gaussion_003 = ski.random_noise(img, mode="gaussian", seed=None, clip=True, mean=0, var=0.006)
        img_gaussion_004 = ski.random_noise(img, mode="gaussian", seed=None, clip=True, mean=0, var=0.007)
        img_gaussion_001 *= 255
        img_gaussion_002 *= 255
        img_gaussion_003 *= 255
        img_gaussion_004 *= 255
        cv2.imwrite(output_dir + fix_name + "13" + filename, img_gaussion_001)
        cv2.imwrite(output_dir + fix_name + "14" + filename, img_gaussion_002)
        cv2.imwrite(output_dir + fix_name + "15" + filename, img_gaussion_003)
        cv2.imwrite(output_dir + fix_name + "16" + filename, img_gaussion_004)


    with open("third.txt", "a+") as f:
        for path, label in files:
            f.write("gaussian_img/" + fix_name + "13" + str(path) + '\t' + str(label) + '\n')
            f.write("gaussian_img/" + fix_name + "14" + str(path) + '\t' + str(label) + '\n')
            f.write("gaussian_img/" + fix_name + "15" + str(path) + '\t' + str(label) + '\n')
            f.write("gaussian_img/" + fix_name + "16" + str(path) + '\t' + str(label) + '\n')




if __name__ == '__main__':
    main()