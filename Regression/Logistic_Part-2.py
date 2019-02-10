import math
import csv
import random

def load_dataset(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def str_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i][j] = float(dataset[i][j])

def minMaxDataset(dataset):
    minMax = []
    for i in range(len(dataset[0])):
        col = [row[i] for row in dataset]
        max_val = max(col)
        min_val = min(col)
        minMax.append([min_val, max_val])
    
    return minMax

def normalization(minMax, dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            numer = dataset[i][j] - minMax[j][0]
            denom = minMax[j][1] - minMax[j][0]
            dataset[i][j] = numer / denom

def prediction(row, coef):
    x = coef[0]
    for i in range(len(row) - 1):
        x += coef[i+1] * row[i]
    return 1 / (1 + math.exp(-x))

def accuracy_score(predicted, actual):
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            count += 1
    
    return (count / len(predicted)) * 100

def cross_validation(dataset,k=5):
    n = len(dataset)
    fold_size = int(n/k)
    folds = []
    dataset_copy = list(dataset)
    for i in range(k):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(0,len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        folds.append(fold)
    return folds

def evaluate_algorithm(dataset, algorithm, alpha, epochs):
    folds = cross_validation(dataset)
    scores = []
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        train = sum(train,[])
        test = []
        for row in fold:
            rowCopy = list(row)
            rowCopy[-1] = None
            test.append(rowCopy)
            
        prediction = algorithm(train, test, alpha, epochs)
        actual = [row[-1] for row in fold]
        score = accuracy_score(prediction, actual)
        scores.append(score)
    return scores

def sgd_logistic(dataset,alpha,epochs):
    b = [0] * len(dataset[0])
    for i in range(epochs):
        for row in dataset:
            y_cap = prediction(row, b)
            b[0] = b[0] + alpha * (row[-1] - y_cap) * y_cap * (1 - y_cap)
            for j in range(len(row) - 1):
                b[j+1] = b[j+1] + alpha * (row[-1] - y_cap) * y_cap * (1 - y_cap) * row[j]
    return b

def logisticRegression(train, test, alpha, epochs):
    coef = sgd_logistic(train, alpha, epochs)
    predictions = []
    for row in test:
        y_pred = prediction(row,coef)
        predictions.append(round(y_pred))
    return predictions

path = 'pima-indians-diabetes.data.csv'
dataset = load_dataset(path)
str_to_float(dataset)
minMax = minMaxDataset(dataset)
normalization(minMax,dataset)
alpha = 0.01
epochs = 1000
scores = evaluate_algorithm(dataset, logisticRegression, alpha, epochs)
print(scores)
avgAccuracy = sum(scores) / len(scores)
print(avgAccuracy)