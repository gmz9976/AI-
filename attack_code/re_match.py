import re

test_file = []



with open("val_list.txt","r") as f:
    for line in f.readlines():
        label, img_name = line.split()
        with open("test_list.txt", 'r') as tf:
            for ll in tf.readlines():
                if re.search(img_name, ll):
                    print(ll)
                    test_file.append(ll)

with open("small.txt", "w") as f:
    for l in test_file:
        f.write(l)
