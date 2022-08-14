import os
import sys
import glob
import cv2

path = "/home/mribeiro/tese/Yolact++/TestSets/Seagull3D_masks"

image_list = sorted(glob.glob(path + '/*.png'))

for image in image_list:
    layer = cv2.imread(image)
    layer = cv2.threshold(layer, 128, 255, cv2.THRESH_BINARY)[1]
    print(os.path.join(path, image))
    cv2.imwrite(os.path.join(path, image), layer)   