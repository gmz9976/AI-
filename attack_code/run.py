import os

cmd = "python attack_second.py --tt "

for i in range(0, 8):
    cmd += str(i)
    os.system(cmd)
    print("第{}个已完成".format(i))
    cmd = cmd.replace(str(i), '')
