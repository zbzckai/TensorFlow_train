# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:00:09 2018

@author: liyingying
"""
import keras.preprocessing.image as im
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def CreateFileandDataGen(OriginalImageDir, GeneratorImageDir):

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
            times = 2000 // filenum + 1
        dirpath = root
        path1 = dirpath.replace(OriginalImageDir, GeneratorImageDir)
        for file in files:##若file为空则不会进入循环。
            img = load_img(root + '\\' + file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=path1 + '\\', save_prefix=os.path.splitext(file)[0],
                                      save_format='jpg'):
                i += 1
                if i > times:
                    break
        for dir in dirs:
            dir1 = path1 + '\\' + dir
            isExists = os.path.exists(dir1)
            if not isExists:
                os.makedirs(dir1)


CreateFileandDataGen('E:\\20181102', 'E:\\20181102new')



