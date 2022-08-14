import os
import sys
import glob
import cv2
from sklearn.metrics import jaccard_score


def IoU(groundtruth_path, predictions_path):
    class String(object):
        def __init__(self, string):
            self.string = string

        def __sub__(self, other):
            if self.string.startswith(other.string):
                return self.string[len(other.string):]

        def __str__(self):
            return self.string

    groundtruth_list = sorted(glob.glob(groundtruth_path + '/*.png'))
    predictions_list = sorted(glob.glob(predictions_path + '/*.png'))

    iou = 0
    iou_final = 0
    iou_number = 0
    number_masks = 0

    for prediction in predictions_list:

        image_name = String(prediction) - String(predictions_path + '/')
        y_pred = cv2.imread(prediction, cv2.IMREAD_GRAYSCALE)

        retval = cv2.haveImageReader(os.path.join(groundtruth_path, image_name))
        if retval:
            y_true = cv2.imread(os.path.join(groundtruth_path, image_name), cv2.IMREAD_GRAYSCALE)
            
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            iou = jaccard_score(y_true, y_pred, pos_label = 255)

            iou_final = iou_final + iou
            iou_number = iou_number +1 

            if iou > 0:
                number_masks = number_masks +1
    
    
    iou_final = iou_final/iou_number
    return iou_final, number_masks

  
