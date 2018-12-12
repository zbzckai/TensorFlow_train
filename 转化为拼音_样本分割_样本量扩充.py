# -*- coding: utf-8 -*-
# @Time    : 2018\11\5 0005 13:49
# @Author  : 凯
# @File    : xunlianceshi.py
import os
from random import randint, sample
import keras.preprocessing.image as im
import math
os.chdir('E:/work/20181102train')
from xpinyin import Pinyin
p = Pinyin()
wj_name = os.listdir('./data2')
wj_name_pinyin = []
for name_i in wj_name:
    wj_name_pinyin.append(p.get_pinyin(name_i,splitter='_'))
train_path = './train'
test_path = './test'

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
##分割训练测试样本，1:9，将中文编程拼音
for kunchong_name,file_pinyin in zip(wj_name,wj_name_pinyin):
    if os.path.exists(train_path+'/'+ file_pinyin):
        continue
    files = os.listdir(r'./data2/{}'.format(kunchong_name))
    files = [x for x in files if 'jpg'  in x]
    ##num_sample = math.ceil(len(files) / 5)
    num_sample = 0  ##b不区分训练测试
    sam_list = sample(range(len(files)),num_sample)
    for index,file in zip(range(len(files)),files):
        train_tmp_path = train_path+'/'+ file_pinyin
        test_tmp_path = test_path+'/'+file_pinyin
        if not os.path.exists(train_tmp_path):
            os.makedirs(train_tmp_path)
        if not os.path.exists(test_tmp_path):
            os.makedirs(test_tmp_path)
        tu = im.load_img(r'./data2/{}/{}'.format(kunchong_name,file))
        if index in sam_list:
            im.save_img(test_tmp_path + '/' + file, tu)
        else:
            im.save_img(train_tmp_path + '/' + file, tu)


import keras.preprocessing.image as im
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def CreateFileandDataGen(OriginalImageDir, GeneratorImageDir,num_zengjia):
    #OriginalImageDir = 'E:\\work\\20181102train\\train'
    #GeneratorImageDir = 'E:\\work\\20181102train\\data\\insect'
    #num_zengjia = 2000
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    GeneratorImageDir = GeneratorImageDir.rstrip("\\")
    OriginalImageDir = OriginalImageDir.rstrip("\\")
    isExists = os.path.exists(GeneratorImageDir)
    if not isExists:
        os.makedirs(GeneratorImageDir)

    for root, dirs, files in os.walk(OriginalImageDir):
        print('============')
        print(root)
        print('============')
        print(dirs)
        print('============')
        print(files)
        times = 50
        filenum = 0
        for fn in os.listdir(root):
            filenum += 1
        if (filenum != 0):
            times = num_zengjia // filenum + 1
        dirpath = root
        print(dirpath)
        path1 = dirpath.replace(OriginalImageDir, GeneratorImageDir)
        print(path1)
        if os.path.exists(path1):
            continue
        os.makedirs(path1)
        for file in files:##若file为空则不会进入循环。
            img = load_img(root + '\\' + file)
            print('shuju')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            print(file)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=path1 + '\\', save_prefix=os.path.splitext(file)[0],
                                      save_format='jpg'):
                i += 1
                if i > times:
                    break
            print('eee')
        for dir in dirs:
            dir1 = path1 + '\\' + dir
            isExists = os.path.exists(dir1)
            if not isExists:
                os.makedirs(dir1)

CreateFileandDataGen('E:\\work\\20181102train\\train', 'E:\\work\\20181102train\\data\\insect',1000)
CreateFileandDataGen('E:\\work\\20181102train\\test', 'E:\\work\\20181102train\\data\\test',200)

###CreateFileandDataGen('E:\\work\\20181102train\\test', 'E:\\work\\20181102train\\data\\test')


