import os
import sys
import glob
import cv2
from sklearn.metrics import jaccard_score

class String(object):
    def __init__(self, string):
        self.string = string

    def __sub__(self, other):
        if self.string.startswith(other.string):
            return self.string[len(other.string):]

    def __str__(self):
        return self.string

groundtruth_path = '/home/mribeiro/tese/3DCRF/TrainSyntheticDataVS/GTSyntheticData3DCRF'
predictions_path = '/home/mribeiro/tese/Yolact++/TestSets/ImagessyntheticData_masks'

groundtruth_list = sorted(glob.glob(groundtruth_path + '/*.png'))
predictions_list = sorted(glob.glob(predictions_path + '/*.png'))

choice = 'Blender'
choice = 'Seagull&Airbus'

iou = 0
iou_final = 0
iou_number = 0
enter = 0
enteriou = 0
if choice == 'Blender':
    for grounftruth in groundtruth_list:

        image_name = String(grounftruth) - String(groundtruth_path + '/')
        
        y_true = cv2.imread(grounftruth, cv2.IMREAD_GRAYSCALE)
        retval = cv2.haveImageReader(os.path.join(predictions_path, image_name))
        if retval:
            y_pred = cv2.imread(os.path.join(predictions_path, image_name), cv2.IMREAD_GRAYSCALE)
            
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            iou = jaccard_score(y_true, y_pred, pos_label = 255)
            enter = enter + 1
            if iou > 0.7:
                enteriou=enteriou+1
                iou_final = iou_final + iou
                iou_number = iou_number +1
            else:
                print(image_name)
        
    iou_final = iou_final/iou_number
    print(iou_final)
    print(iou_number, enter, enteriou)

elif choice == 'Seagull&Airbus':
    for prediction in predictions_list:

        image_name = String(prediction) - String(predictions_path + '/')
        y_pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)
        
        retval = cv2.haveImageReader(os.path.join(groundtruth_path, image_name))
        if retval:
            y_true = cv2.imread(os.path.join(groundtruth_path, image_name), cv2.IMREAD_GRAYSCALE)
            
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            iou = jaccard_score(y_true, y_pred, pos_label = 255)

            #if iou > 0.5:
            iou_final = iou_final + iou
            iou_number = iou_number +1
        
    iou_final = iou_final/iou_number
    print(iou_final)
    print(iou_number)
