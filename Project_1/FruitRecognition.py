import os, cv2
import numpy as np

# dataset link
# https://www.kaggle.com/moltean/fruits/version/44

path = 'fruits/'
imgPath = []
for root, folder, files in os.walk(path):
	for file in files:
		imgPath.append(root+'/'+file)

imgArray = []
for img in imgPath:
    gray = cv2.imread(img, cv2.COLOR_BGR2GRAY)
    imgArray.append(gray)

imgArray = np.asarray(imgArray)
imgArray.reshape(3515, 100*100*3)
names = os.listdir(path)
labels = np.zeros(imgArray.shape[0])

def dist(x1,x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def knn(train, x):
    m = train.shape[0]
    distance = []
    for i in range(m):
        distance.append(dist(train[i], x))
    distance = np.asarray(distance)
    indexes = np.argsort(distance)
    sortedLabels = labels[indexes][:5]
