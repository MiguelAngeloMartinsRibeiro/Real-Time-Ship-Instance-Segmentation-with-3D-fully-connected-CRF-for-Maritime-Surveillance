from PIL import Image
import numpy as np
import os
import sys
import glob

path = '/home/mribeiro/tese/Yolact++/data/datasets/TestSets/Seagull3DAnnotations'

path_list = sorted(glob.glob(path + '/*.png'))

for image in path_list:

    img = Image.open(image).convert('RGB')
    
    width = img.size[0] 
    height = img.size[1] 
    for i in range(0,width):# process all pixels
        for j in range(0,height):
            data = img.getpixel((i,j))
            #print(data) #(255, 255, 255)
            if (data[0]==128 and data[1]==0 and data[2]==0):
                img.putpixel((i,j),(255, 255, 255))
    img.save(image)