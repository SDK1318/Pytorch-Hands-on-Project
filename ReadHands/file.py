import os
import shutil

dir_path = "D:/Code/ReadHands/GestureC/9"
num = 1859
if __name__ == '__main__':
    for file in os.listdir(dir_path):
        s = '%0d'% num
        os.rename(os.path.join(dir_path, file), os.path.join(dir_path, str(s) + '.jpg'))
        num += 1