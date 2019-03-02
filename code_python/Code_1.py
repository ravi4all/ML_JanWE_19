import os
import numpy as np
from skimage import io, color

trainingPath = 'dataset/Training/'
testPath = 'dataset/Test/'

def readPath(path):
    imgPath = []
    for root, folder, files in os.walk(path):
        for file in files:
            imgPath.append(root+'/'+file)

    return imgPath

def convertArray(imgPath):
    imgArray = []
    for img in imgPath:
        image = color.rgb2gray(io.imread(img))
        imgArray.append(image)
    return imgArray

trainingImagesPath = readPath(trainingPath)
trainingImgArray = convertArray(trainingImagesPath)

testImagesPath = readPath(testPath)
testImgArray = convertArray(testImagesPath)

trainingImgArray = np.asarray(trainingImgArray)
testImgArray = np.asarray(testImgArray)
np.save('trainFruits.npy',trainingImgArray)
np.save('testFruits.npy',testImgArray)