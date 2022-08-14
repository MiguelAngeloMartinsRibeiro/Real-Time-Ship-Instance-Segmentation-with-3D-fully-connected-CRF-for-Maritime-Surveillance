import numpy as np
import argparse
import cv2
import os
from IoUCalculator import IoU
from numbers import Number
from itertools import product
import pydensecrf.densecrf as dcrf
import time
import pickle

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

def Train2D(imagesPath, masksPath, GTPath, outputPath, mode):

    sXYgaussianCombo = [1, 3, 5, 10, 20]
    gaussianCompCombo = [1, 3, 5, 10, 20]
    sXYbilateralCombo = [5, 20, 50, 80]
    sRGBCombo = [1, 5, 13, 30, 75]
    bilateralCompCombo = [1, 3, 5, 10]

    #sXYgaussianCombo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #gaussianCompCombo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #sXYbilateralCombo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #sRGBCombo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #bilateralCompCombo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    gtProb = 0.6
    nLabels = 2
    images, imageNames = loadImages(imagesPath)
    masks = loadMasks(masksPath)
    sizeOfImages = [images[0].shape[0], images[0].shape[1], len(images)]

    BilateralCombos = list(product(sXYbilateralCombo, sRGBCombo))
    feats2Dgaussian = []
    featsImgBilateral = []
    feats2Dbilateral = []
    U = []

    if(mode == "write"):
        
        for i in range(sizeOfImages[2]):
            Uaux = makeUnaryfromMasks(masks, gtProb, nLabels, maskNumber = i)
            U.append(Uaux)

            with open('UTest.txt', 'wb') as f:
                pickle.dump(U, f)
        
        for aux in sXYgaussianCombo:
            sXYgaussian = [aux, aux]
            feats2D = createPairwiseGaussian2D(sXYgaussian[0], sXYgaussian[1], shape = sizeOfImages)
            feats2Dgaussian.append(feats2D)

        with open('feats2DgaussianTest.txt', 'wb') as g:
            pickle.dump(feats2Dgaussian, g)
        
        for i in range(sizeOfImages[2]):
            for sXYbilateralCombo, sRGBCombo in BilateralCombos:
            
                sXYbilateral = [sXYbilateralCombo, sXYbilateralCombo]
                sRGB = [sRGBCombo, sRGBCombo, sRGBCombo]

                feats2D = createPairwiseBilateral2D(sXYbilateralCombo[0], sXYbilateral[1], sRGB[0], sRGB[1], sRGB[2], img = images[i], shape = sizeOfImages)
                feats2Dbilateral.append(feats2D)
            featsImgBilateral.append(feats2Dbilateral)
            feats2Dbilateral.clear()

        with open("featsBilateralImgTest.txt", "wb") as h:
            pickle.dump(featsImgBilateral, h)
        
    elif(mode == "read"):

        with open('feats2DgaussianTest.txt', 'rb') as f:
            feats2Dgaussian = pickle.load(f)
        with open('featsBilateralImgTest.txt', 'rb') as g:
            featsImgBilateral = pickle.load(g)
        with open('UTest.txt', 'rb') as h:
            U = pickle.load(h)

    allCombos = list(product(sXYgaussianCombo, gaussianCompCombo, sXYbilateralCombo, sRGBCombo, bilateralCompCombo))
    
    for sXYgaussianCombo, gaussianCompCombo, sXYbilateralCombo, sRGBCombo, bilateralCompCombo in allCombos:

        sXYgaussian = [sXYgaussianCombo, sXYgaussianCombo, sXYgaussianCombo]
        gaussianComp = gaussianCompCombo
        sXYbilateral = [sXYbilateralCombo, sXYbilateralCombo, sXYbilateralCombo]
        sRGB = [sRGBCombo, sRGBCombo, sRGBCombo]
        bilateralComp = bilateralCompCombo
        
        for i in range(sizeOfImages[2]):

            crf2d = dcrf.DenseCRF2D(sizeOfImages[0], sizeOfImages[1], nLabels)
            crf2d.setUnaryEnergy(U[i])
            
            feats2D = createPairwiseGaussian2D(sXYgaussian[0], sXYgaussian[1], shape = sizeOfImages)
            crf2d.addPairwiseEnergy(feats2D, compat=gaussianComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            
            feats2D = createPairwiseBilateral2D(sXYbilateral[0], sXYbilateral[1], sRGB[0], sRGB[1], sRGB[2], img = images[i], shape = sizeOfImages)
            crf2d.addPairwiseEnergy(feats2D, compat=bilateralComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
            Q = crf2d.inference(5)
            MAP = np.argmax(Q, axis=0)
            
            cv2.imwrite(os.path.join(outputPath , imageNames[i]), (MAP*255).reshape(sizeOfImages[0], sizeOfImages[1]))
            

        iou, number = IoU(GTPath ,outputPath)
        print(sXYgaussianCombo, gaussianCompCombo, sXYbilateralCombo, sRGBCombo, bilateralCompCombo, number, iou)
                   
def main():

    masksPath = "/home/mribeiro/tese/3DCRF/TinyTest/PredictionsYolact"
    imagesPath = "/home/mribeiro/tese/3DCRF/TinyTest/ImagesSyntheticData"
    GTPath = "/home/mribeiro/tese/3DCRF/TinyTest/GTSyntheticData3DCRF"
    outputPath = "/home/mribeiro/tese/3DCRF/TinyTest/Output2D"
    iou, number = IoU(GTPath ,masksPath)
    mode = "read"
    print("IoU before CRF")
    print(iou, number)

    Train2D(imagesPath, masksPath, GTPath, outputPath, mode)

if __name__ == "__main__":
    main()


