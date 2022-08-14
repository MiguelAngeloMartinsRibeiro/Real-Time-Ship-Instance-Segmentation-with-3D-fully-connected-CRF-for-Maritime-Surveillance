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
        img = img.flatten()
        if img is not None:
            images = np.concatenate([images,img])
            imageNames.append(filename)

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
    featsgaussian = np.zeros((feat_size, shape[0] * shape[1] * shape[2]), dtype=np.float32)

    for z in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                featsgaussian[0, (z*shape[0]*shape[1] + i*shape[1]+j)] = i / sx
                featsgaussian[1, (z*shape[0]*shape[1] + i*shape[1]+j)] = j / sy
                featsgaussian[2, (z*shape[0]*shape[1] + i*shape[1]+j)] = z / sz
    return featsgaussian

def createPairwiseBilateral3Dstatic(sx, sy, sz, img, shape):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    feat_size = 6
    featsbilateral = np.zeros((feat_size, shape[0] * shape[1] * shape[2]), dtype=np.float32)
    for z in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                featsbilateral[0, (z*shape[0]*shape[1] + i*shape[1]+j)] = i / sx
                featsbilateral[1, (z*shape[0]*shape[1] + i*shape[1]+j)] = j / sy
                featsbilateral[2, (z*shape[0]*shape[1] + i*shape[1]+j)] = z / sz
    return featsbilateral

def createPairwiseBilateral3Dvariable(sr, sg, sb, img, shape, featsbilateral):
    """
    A simple reference implementation for the 2D case. The ND implementation
    is faster.
    """
    for z in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                featsbilateral[3, (z*shape[0]*shape[1] + i*shape[1]+j)] =  img[(j+i*shape[1] + z*shape[0]*shape[1])*3+0]/ sr
                featsbilateral[4, (z*shape[0]*shape[1] + i*shape[1]+j)] =  img[(j+i*shape[1] + z*shape[0]*shape[1])*3+1]/ sg
                featsbilateral[5, (z*shape[0]*shape[1] + i*shape[1]+j)] =  img[(j+i*shape[1] + z*shape[0]*shape[1])*3+2]/ sb
    return featsbilateral

def main():
    
    masksPath = "/home/mribeiro/tese/3DCRF/TinyTest30/PredictionsYolact"
    imagesPath = "/home/mribeiro/tese/3DCRF/TinyTest30/ImagesSyntheticData"
    GTPath = "/home/mribeiro/tese/3DCRF/TinyTest30/GTSyntheticData3DCRF"
    outputPath = "/home/mribeiro/tese/3DCRF/TinyTest30/Output3D"
    start_time = time.time()
    #######################################################################################
    #             STATIC VARIABLES
    #######################################################################################
    gtProb = 0.7
    nLabels = 2
    images, imageNames = loadImages(imagesPath)
    masks = loadMasks(masksPath)

    loadingTime = time.time()

    sizeOfImages = [550,550,1] #[images[0].shape[0], images[0].shape[1], len(images)]
    sXYZgaussian = [1, 1, 1]
    gaussianComp = 1
    sXYZbilateral = [5, 5, 5]
    sRGB = [3, 3, 3]
    bilateralComp = 5

    U = makeUnaryfromMasks(masks, gtProb, nLabels, sizeOfImages)
    crf3d = dcrf.DenseCRF3D(sizeOfImages[0], sizeOfImages[1], sizeOfImages[2], nLabels)
    crf3d.setUnaryEnergy(U)

    unaryTime = time.time()
    featsgaussian = createPairwiseGaussian3D(sXYZgaussian[0], sXYZgaussian[1], sXYZgaussian[2], shape=sizeOfImages[:3])
    crf3d.addPairwiseEnergy(featsgaussian, compat=gaussianComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    gaussianTime = time.time()
    featsbilateralxyz = createPairwiseBilateral3Dstatic(sXYZbilateral[0], sXYZbilateral[1], sXYZbilateral[2], img=images, shape = sizeOfImages)
    staticTime = time.time()
    #######################################################################################
    #             Calculated in every image
    #######################################################################################

    featsbilateralimg = createPairwiseBilateral3Dvariable(sRGB[0], sRGB[1], sRGB[2], img=images, shape = sizeOfImages, featsbilateral=featsbilateralxyz)
    crf3d.addPairwiseEnergy(featsbilateralimg, compat=bilateralComp, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    bilateralTime = time.time()
    Q = crf3d.inference(4)
    inferenceTime = time.time()
    
    MAP = np.argmax(Q, axis=0)

    for i in range(sizeOfImages[2]):
        MAPaux = MAP[(sizeOfImages[0]*sizeOfImages[1])*i:((sizeOfImages[0]*sizeOfImages[1])*(i+1))]
        cv2.imwrite(os.path.join(outputPath , imageNames[i]), (MAPaux*255).reshape(sizeOfImages[0], sizeOfImages[1]))
    savingTime = time.time()

    print((loadingTime-start_time), (unaryTime-loadingTime), (gaussianTime-unaryTime), (staticTime-start_time), (bilateralTime-staticTime), (inferenceTime-bilateralTime), (savingTime-bilateralTime), (savingTime-start_time))
    
if __name__ == "__main__":
    main()