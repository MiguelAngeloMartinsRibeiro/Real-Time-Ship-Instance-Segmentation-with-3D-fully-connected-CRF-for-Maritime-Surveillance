import numpy as np
import argparse
import cv2
import os
from IoUCalculator import IoU
from numbers import Number
from itertools import product
import pydensecrf.densecrf as dcrf
import time

########### Load Masks #################################################################################################
# Input path of masks
# Reads masks and turns them into binary masks (0,1). The original range is (0-255)
# Inserts every single mask into a list. They are organized alphabetically
# The output is that list
########################################################################################################################
def loadMasks(folder):
    masks = []
    for filename in sorted(os.listdir(folder)):
        mask = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
        if mask is not None:
            mask[mask < 255] = 0
            mask[mask == 255] = 1
            masks.append(mask)
    return masks


########### Load Images #################################################################################################
# Input path of images
# Reads images and their names
# Inserts every single image and image name into a list. They are organized alphabetically
# The output is a list of images and a list of images names
#########################################################################################################################
def loadImages(folder):
    images = []
    imageNames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            imageNames.append(filename)
            images.append(img)

    return images, imageNames

def makeUnaryfromMasks(masks, gtProb, nLabels, maskNumber):

    assert 0 < gtProb < 1

    mask = masks[maskNumber]
    mask = mask.flatten()
    labels = mask

    foregroundEnergy = -np.log(1.0 - gtProb)
    backgroundEnergy = -np.log(gtProb)
    
    U = np.full((nLabels, len(labels)), labels, dtype='float32')
    
    U[0, labels == 0] = backgroundEnergy
    U[0, labels == 1] = foregroundEnergy
    U[1, labels == 0] = foregroundEnergy
    U[1, labels == 1] = backgroundEnergy

    return U

def createPairwiseGaussian2D(sx, sy, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 2
    feats = np.zeros((feat_size, shape[0], shape[1]), dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            feats[0, i, j] = i / sx
            feats[1, i, j] = j / sy
    return feats.reshape([feat_size, -1])

def createPairwiseBilateral2D(sx, sy, sr, sg, sb, img, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 5
    feats = np.zeros((feat_size, shape[0], shape[1]), dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            feats[0, i, j] = i / sx
            feats[1, i, j] = j / sy
            feats[2, i, j] = img[i, j, 0] / sr
            feats[3, i, j] = img[i, j, 1] / sg
            feats[4, i, j] = img[i, j, 2] / sb
    return feats.reshape([feat_size, -1])

def Train2D(imagesPath, masksPath, GTPath, outputPath):

    sXYZgaussianCombo = [1, 2]
    gaussianCompCombo = [1, 2]
    sXYZbilateralCombo = [1, 2, 3, 4, 5]
    sRGBCombo = [1, 2, 3, 4, 5]
    bilateralCompCombo = [1, 2, 3, 4, 5]

    gtProb = 0.6
    nLabels = 2
    images, imageNames = loadImages(imagesPath)
    masks = loadMasks(masksPath)

    sizeOfImages = [images[0].shape[0], images[0].shape[1], len(images)]

    allCombos = list(product(sXYZgaussianCombo, gaussianCompCombo, sXYZbilateralCombo, sRGBCombo, bilateralCompCombo))
    
    for sXYZgaussianCombo, gaussianCompCombo, sXYZbilateralCombo, sRGBCombo, bilateralCompCombo in allCombos:

        sXYZgaussian = [sXYZgaussianCombo, sXYZgaussianCombo, sXYZgaussianCombo]
        gaussianComp = gaussianCompCombo
        sXYZbilateral = [sXYZbilateralCombo, sXYZbilateralCombo, sXYZbilateralCombo]
        sRGB = [sRGBCombo, sRGBCombo, sRGBCombo]
        bilateralComp = bilateralCompCombo
        
        for i in range(sizeOfImages[2]):

            U = makeUnaryfromMasks(masks, gtProb, nLabels, maskNumber = i)

            crf2d = dcrf.DenseCRF2D(sizeOfImages[0], sizeOfImages[1], nLabels)
            crf2d.setUnaryEnergy(U)
            
            feats2D = createPairwiseGaussian2D(sXYZgaussian[0], sXYZgaussian[1], shape = sizeOfImages)
            crf2d.addPairwiseEnergy(feats2D, compat=gaussianComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            
            feats2D = createPairwiseBilateral2D(sXYZbilateral[0], sXYZbilateral[1], sRGB[0], sRGB[1], sRGB[2], img = images[i], shape = sizeOfImages)
            crf2d.addPairwiseEnergy(feats2D, compat=bilateralComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
            Q = crf2d.inference(5)
            MAP = np.argmax(Q, axis=0)
            
            cv2.imwrite(os.path.join(outputPath , imageNames[i]), (MAP*255).reshape(sizeOfImages[0], sizeOfImages[1]))
            

        iou, number = IoU(GTPath ,outputPath)
        print(sXYZgaussianCombo, gaussianCompCombo, sXYZbilateralCombo, sRGBCombo, bilateralCompCombo, number, iou)

def main():

    masksPath = "/home/mribeiro/tese/3DCRF/Seagull3D_2DCRF/PredictionsYolact"
    imagesPath = "/home/mribeiro/tese/3DCRF/Seagull3D_2DCRF/ImagesSyntheticData"
    GTPath = "/home/mribeiro/tese/3DCRF/Seagull3D_2DCRF/GTSyntheticData3DCRF"
    outputPath = "/home/mribeiro/tese/3DCRF/Seagull3D_2DCRF/Output2D"
    iou, number = IoU(GTPath ,masksPath)
    print("IoU before CRF")
    print(iou, number)
    option = "Test"
    option = "Train"

    if option == "Train":

        Train2D(imagesPath, masksPath, GTPath, outputPath)

    elif option == "Test":

        gtProb = 0.6
        nLabels = 2
        images, imageNames = loadImages(imagesPath)
        masks = loadMasks(masksPath)

        sizeOfImages = [images[0].shape[0], images[0].shape[1], len(images)]
        sXYZgaussian = [2, 2, 2]
        gaussianComp = 1
        sXYZbilateral = [5, 5, 5]
        sRGB = [5, 5, 5]
        bilateralComp = 5

        for i in range(sizeOfImages[2]):

            U = makeUnaryfromMasks(masks, gtProb, nLabels, maskNumber = i)
            crf2d = dcrf.DenseCRF2D(sizeOfImages[0], sizeOfImages[1], nLabels)
            crf2d.setUnaryEnergy(U)
            
            feats2D = createPairwiseGaussian2D(sXYZgaussian[0], sXYZgaussian[1], shape = sizeOfImages)
            crf2d.addPairwiseEnergy(feats2D, compat=gaussianComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

            feats2D = createPairwiseBilateral2D(sXYZbilateral[0], sXYZbilateral[1], sRGB[0], sRGB[1], sRGB[2], img = images[i], shape = sizeOfImages)
            crf2d.addPairwiseEnergy(feats2D, compat=bilateralComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            start_time = time.time()
            Q = crf2d.inference(4)
            MAP = np.argmax(Q, axis=0)
            
            cv2.imwrite(os.path.join(outputPath , imageNames[i]), (MAP*255).reshape(sizeOfImages[0], sizeOfImages[1]))

        iou, number = IoU(GTPath ,outputPath)
        print("IoU after CRF")
        print(sXYZgaussian, gaussianComp, sXYZbilateral, bilateralComp, sRGB, number, iou)
    else:
        print("Please Choose an Option Train or Test")
        return(0)

if __name__ == "__main__":
    main()


