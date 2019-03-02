import os, cv2
import numpy as np
from skimage import io, color

trainingData = np.load('trainFruits.npy')
trainingData = trainingData.reshape(trainingData.shape[0], -1)

labels = os.listdir('dataset/Training/')
# print(labels)
labelName = {i : labels[i] for i in range(len(labels))}
# print(labelName)

labelLength = []
for root, folder, files in os.walk('dataset/Training/'):
    labelLength.append(len(files))
# print(labelLength)

outputLabels = np.zeros((len(trainingData), 1))
slice_1 = 0
slice_2 = 0
try:
    for i in range(len(labelLength)):
        slice_1 += labelLength[i]
        slice_2 += labelLength[i+1]
        print(slice_1, slice_2)
        outputLabels[slice_1:slice_2, :] = i
except BaseException:
    pass

def distance(x1,x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def knn(x, train,k):
#     we need to calculate distance of input image with each image of
#     traing data, so we find out shape of training data
    n = train.shape[0]
    dist = []
    for i in range(n):
#         we calculate distance b/w out input image and training image
#         and append the distances in a list
        dist.append(distance(x, train[i]))
#     convert distance list into numpy array
    dist = np.asarray(dist)
#     sort the distance to get nearest or minimum distances
#     so we use argsort which will give indexes of sorted data
    indexes = np.argsort(dist)[:k]
    sortedLables = outputLabels[indexes]
    counts = np.unique(sortedLables, return_counts=True)
#     ([0,1,3], [3,3,4])
    return counts[0][np.argmax(counts[1])]

# font = cv2.FONT_HERSHEY_SIMPLEX
# cam = cv2.VideoCapture(0)
# while True:
#    ret, frame = cam.read()
#    if ret == True:
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        roi = frame[100:300, 100:300,:]
#        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
#        obj = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#        fruit = cv2.resize(obj, (100, 100))
#        # print(fruit.shape)
#        lab = knn(fruit.flatten(), trainingData, 5)
#        text = labels[int(lab)]
#        cv2.putText(frame, text, (100, 100), font, 1, (255, 255, 0), 2)
#        cv2.imshow('fruit recognition', frame)
#        # cv2.imshow('fruit',roi)
#        k = cv2.waitKey(33) & 0xFF
#        if k == 27:
#            break
#    else:
#        print('Error')
#
# cam.release()
# cv2.destroyAllWindows()