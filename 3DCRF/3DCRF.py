import numpy as np
import argparse
import cv2
import os
from IoUCalculator import IoU
from numbers import Number
from pydensecrf.utils import unary_from_labels
from itertools import product
import pydensecrf.densecrf as dcrf
import time
import sys

########### Load Masks #################################################################################################
# Input path of masks
# Reads masks and turns them into binary masks (0,1). The original range is (0-255)
# Inserts every single mask into a list. They are organized alphabetically
# The output is that list
########################################################################################################################
def loadMasks(folder):
    masks = []
    for filename in sorted(os.listdir(folder)):
        #mask = cv2.imread(os.path.join(folder,filename))
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

def makeUnaryfromMasks(masks, gtProb, nLabels, sizeOfImages):

    assert 0 < gtProb < 1
    
    for i in range(sizeOfImages[2]):
        mask = masks[i]
        mask = mask.flatten()
        if i == 0:
            labels = mask
        else:
            labels = np.append(labels, mask)

    foregroundEnergy = -np.log(1.0 - gtProb)
    backgroundEnergy = -np.log(gtProb)
    
    U = np.full((nLabels, len(labels)), labels, dtype='float32')
    
    U[0, labels == 0] = backgroundEnergy
    U[0, labels == 1] = foregroundEnergy
    U[1, labels == 0] = foregroundEnergy
    U[1, labels == 1] = backgroundEnergy

    return U

def createPairwiseGaussian3D(sx, sy, sz, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 3
    feats = np.zeros((feat_size, shape[0] * shape[1] * shape[2]), dtype=np.float32)

    for z in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                feats[0, (z*shape[0]*shape[1] + i*shape[1]+j)] = i / sx
                feats[1, (z*shape[0]*shape[1] + i*shape[1]+j)] = j / sy
                feats[2, (z*shape[0]*shape[1] + i*shape[1]+j)] = z / sz
    return feats

def createPairwiseBilateral3D(sx, sy, sz, sr, sg, sb, img, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 6
    feats = np.zeros((feat_size, shape[0] * shape[1] * shape[2]), dtype=np.float32)
    for z in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                feats[0, (z*shape[0]*shape[1] + i*shape[1]+j)] = i / sx
                feats[1, (z*shape[0]*shape[1] + i*shape[1]+j)] = j / sy
                feats[2, (z*shape[0]*shape[1] + i*shape[1]+j)] = z / sz
                feats[3, (z*shape[0]*shape[1] + i*shape[1]+j)] = img[z][i, j, 0] / sr
                feats[4, (z*shape[0]*shape[1] + i*shape[1]+j)] = img[z][i, j, 1] / sg
                feats[5, (z*shape[0]*shape[1] + i*shape[1]+j)] = img[z][i, j, 2] / sb
    return feats

def Train3D(imagesPath, masksPath, GTPath, outputPath):
    
    sXYZgaussianCombo = 1
    gaussianCompCombo = 1
    sXYZbilateralCombo = 4
    sRGBCombo = 5
    bilateralCompCombo = 5
    batch = 5

    gtProb = 0.6
    nLabels = 2
    images, imageNames = loadImages(imagesPath)
    masks = loadMasks(masksPath)

    print("loadComplete")

    times = 0

    #allCombos = list(product(sXYZgaussianCombo, gaussianCompCombo, sXYZbilateralCombo, sRGBCombo, bilateralCompCombo))
    #for sXYZgaussianCombo, gaussianCompCombo, sXYZbilateralCombo, sRGBCombo, bilateralCompCombo in allCombos:

    sXYZgaussian = [sXYZgaussianCombo, sXYZgaussianCombo, sXYZgaussianCombo] 
    gaussianComp = gaussianCompCombo
    sXYZbilateral = [sXYZbilateralCombo, sXYZbilateralCombo, sXYZbilateralCombo] 
    sRGB = [sRGBCombo, sRGBCombo, sRGBCombo]
    bilateralComp = bilateralCompCombo

    for k in range(int(len(images)/batch)):
        imagesBatch = images[k*batch:(k+1)*batch]
        masksBatch = masks[k*batch:(k+1)*batch]
        imageNamesBatch = imageNames[k*batch:(k+1)*batch]
        sizeOfImages = [imagesBatch[0].shape[0], imagesBatch[0].shape[1], len(imagesBatch)]

        U = makeUnaryfromMasks(masksBatch, gtProb, nLabels, sizeOfImages)

        crf3d = dcrf.DenseCRF3D(sizeOfImages[0], sizeOfImages[1], sizeOfImages[2], nLabels)
        crf3d.setUnaryEnergy(U)

        feats3D = createPairwiseGaussian3D(sXYZgaussian[0], sXYZgaussian[1], sXYZgaussian[2], shape=sizeOfImages[:3])
        crf3d.addPairwiseEnergy(feats3D, compat=gaussianComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        feats3D = createPairwiseBilateral3D(sXYZbilateral[0], sXYZbilateral[1], sXYZbilateral[2], sRGB[0], sRGB[1], sRGB[2], img=imagesBatch, shape = sizeOfImages)
        crf3d.addPairwiseEnergy(feats3D, compat=bilateralComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = crf3d.inference(4)

        MAP = np.argmax(Q, axis=0)

        for i in range(sizeOfImages[2]):
            MAPaux = MAP[(sizeOfImages[0]*sizeOfImages[1])*i:((sizeOfImages[0]*sizeOfImages[1])*(i+1))]
            cv2.imwrite(os.path.join(outputPath , imageNamesBatch[i]), (MAPaux*255).reshape(sizeOfImages[0], sizeOfImages[1]))
        
        times = times + 1

    iou, number = IoU(GTPath, outputPath)
    print(sXYZgaussianCombo, gaussianCompCombo, sXYZbilateralCombo, sRGBCombo, bilateralCompCombo, number, times, iou)


def main():

    masksPath = "/home/mribeiro/tese/3DCRF/Seagull3D/PredictionsYolact"
    imagesPath = "/home/mribeiro/tese/3DCRF/Seagull3D/ImagesSyntheticData"
    GTPath = "/home/mribeiro/tese/3DCRF/Seagull3D/GTSyntheticData3DCRF"
    outputPath = "/home/mribeiro/tese/3DCRF/Seagull3D/Output3D"
    iou, number = IoU(GTPath ,masksPath)
    print("IoU before CRF")
    print(iou, number)
    option = "Test"
    option = "Train"
    #option = "IoU"

    if option == "Train":

        Train3D(imagesPath, masksPath, GTPath, outputPath)

    elif option == "Test":
        gtProb = 0.6
        nLabels = 2
        images, imageNames = loadImages(imagesPath)
        masks = loadMasks(masksPath)

        sizeOfImages = [images[0].shape[0], images[0].shape[1], len(images)]
        sXYZgaussian = [1,1,1]
        gaussianComp = 1
        sXYZbilateral = [4, 4, 4]
        sRGB = [3, 3, 3]
        bilateralComp = 5
        
        U = makeUnaryfromMasks(masks, gtProb, nLabels, sizeOfImages)
        crf3d = dcrf.DenseCRF3D(sizeOfImages[0], sizeOfImages[1], sizeOfImages[2], nLabels)
        crf3d.setUnaryEnergy(U)

        feats3D = createPairwiseGaussian3D(sXYZgaussian[0], sXYZgaussian[1], sXYZgaussian[2], shape=sizeOfImages[:3])
        crf3d.addPairwiseEnergy(feats3D, compat=gaussianComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats3D = createPairwiseBilateral3D(sXYZbilateral[0], sXYZbilateral[1], sXYZbilateral[2], sRGB[0], sRGB[1], sRGB[2], img=images, shape = sizeOfImages)
        crf3d.addPairwiseEnergy(feats3D, compat=bilateralComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = crf3d.inference(4)
        MAP = np.argmax(Q, axis=0)

        for i in range(sizeOfImages[2]):
            MAPaux = MAP[(sizeOfImages[0]*sizeOfImages[1])*i:((sizeOfImages[0]*sizeOfImages[1])*(i+1))]
            cv2.imwrite(os.path.join(outputPath , imageNames[i]), (MAPaux*255).reshape(sizeOfImages[0], sizeOfImages[1]))
        
        iou, number = IoU(GTPath ,outputPath)
        print("IoU after CRF")

        print(sXYZgaussian, gaussianComp, sXYZbilateral, bilateralComp, sRGB, number, iou)
    elif option == "IoU":
        iou, number = IoU(GTPath, outputPath)
        print("IoU before CRF")
        print(iou, number)
    else:
        print("Please Choose an Option Train or Test")
        return(0)
    
if __name__ == "__main__":
    main()