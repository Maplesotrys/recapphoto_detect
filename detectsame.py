import cv2
import numpy as np
import os
from PIL import Image
import math
import operator
from functools import reduce


srcdir = './data/轿车360'

def compare(pic1,pic2):
    '''
    :param pic1: 图片1路径
    :param pic2: 图片2路径
    :return: 返回对比的结果
    '''
    image1 = Image.open(pic1)
    image2 = Image.open(pic2)

    histogram1 = image1.histogram()
    histogram2 = image2.histogram()

    differ = math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b)**2,histogram1, histogram2)))/len(histogram1))

    return differ
# image1 = cv2.imread(file1)
# image2 = cv2.imread(file2)
# difference = cv2.subtract(image1, image2)
# result = not np.any(difference) #if difference is all zeros it will return False
filelist = os.listdir(srcdir) #列出文件夹下所有的目录与文件
filelist.sort(key=lambda x:int(x[:-4]))
a = filelist[1]
lens = len(filelist)
for i in range(0,lens):
    path = os.path.join(srcdir,filelist[i])
    path = path.replace('\\', '/')
    # cur = cv2.imread(path)
    for j in range(i+1,lens):
        path1 = os.path.join(srcdir,filelist[j])
        path1 = path1.replace('\\', '/')
        # other = cv2.imread(path1)
        if os.path.isfile(path) and  os.path.isfile(path1):
            if compare(path,path1) == 0:
                print(path1)
                os.remove(path1)

