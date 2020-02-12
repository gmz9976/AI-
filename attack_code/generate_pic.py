import os
import shutil


fix_name = "125"
files_path = "./final/v3/v5/007/"
new_path = "./final/false/"


def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files

def find_file(filepath):
    for a, b, c in os.walk(filepath):
        for file in c:
            if file.endswith("png"):
                shutil.copyfile(filepath + file, new_path + fix_name + file)
                #os.rename(filepath + file, filepath + file.replace("001", ""))
                print(new_path + fix_name + file + "已生成")

def main():
    find_file(files_path)
    files = get_original_file(files_path + "val_list.txt")
    print(files)
    with open("six.txt", "a+") as f:
        for path, label in files:
            f.write("false/" + fix_name + str(path) + '\t' + str(label) + '\n')
    # for i in range(len(files)):
    #     files[i][0] = fix_name + files[i][0]
    # file_list = fix_name + "val_list.txt"
    # with open(file_list, "w") as f:
    #     for path, label in files:
    #         f.write(str(label) + '\t' + str(path) + '\n')



if __name__=='__main__':
    main()