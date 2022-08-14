import os
import sys
import glob
import cv2

input_path = "/home/mribeiro/tese/Yolact++/TestSets/Output"
output_path = "/home/mribeiro/tese/Yolact++/TestSets/OutputFinal"

image_list = sorted(glob.glob(input_path + '/*.png'))

class String(object):
    def __init__(self, string):
        self.string = string

    def __sub__(self, other):
        if self.string.startswith(other.string):
            return self.string[len(other.string):]

    def __str__(self):
        return self.string

last_name = 'x'
for image in image_list:
    image_name = String(image)-String(input_path + "/")
    image_name = image_name.split('_')
    image_name = image_name[0]

    if image_name != last_name:
        layer_init = cv2.imread(image)
        layer_init = cv2.cvtColor(layer_init, cv2.COLOR_BGR2GRAY)
        layer_init = cv2.threshold(layer_init, 128, 255, cv2.THRESH_BINARY)[1]
        last_name = image_name
        cv2.imwrite(os.path.join(output_path, image_name + '.png'), layer_init)   
    else:
        layer = cv2.imread(image)
        layer = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        layer = cv2.threshold(layer, 128, 255, cv2.THRESH_BINARY)[1]
        layer = cv2.add(layer, layer_init)
        layer_init = layer
        cv2.imwrite(os.path.join(output_path, image_name + '.png'), layer)
