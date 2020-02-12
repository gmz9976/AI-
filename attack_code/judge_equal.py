import numpy as np
from PIL import Image

input_image_dir = "./0.30/008/n02085782_1039.png"
output_image_dir = "./0.30/014/n02085782_1039.png"

a = np.array(Image.open(input_image_dir))
b = np.array(Image.open(output_image_dir))

print(a.shape)
print(b.shape)

print((a==b).all())