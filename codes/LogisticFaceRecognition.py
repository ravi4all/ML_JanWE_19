import numpy as np
import random
from sklearn.model_selection import train_test_split

face_1 = np.load('face_1.npy').reshape(200,50*50*3)
face_2 = np.load('face_2.npy').reshape(200,50*50*3)

users = {1 : "Ritij", 2 : "Ravi"}

labels = np.zeros((400,1))
labels[:200] = 0.0
labels[200:] = 1.0
dataset = np.concatenate([face_1, face_2])
dataset = np.append(dataset, labels, axis = 1)

def prediction(row, coef):
    x = coef[0]
    for i in range(len(row) - 1):
        x += coef[i + 1] * row[i]
    return 1 / (1 + np.exp(-x))


def accuracy_score(predicted, actual):
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            count += 1

    return (count / len(predicted)) * 100

def evaluate_algorithm(dataset, algorithm, alpha, epochs):
    scores = []
    x_train,x_test,y_train,y_test = train_test_split(dataset[:,:-1], dataset[:,-1], test_size=0.25)
    prediction,coef = algorithm(x_train, x_test, alpha, epochs)
    actual = [labels[i] for i in range(len(labels))]
    score = accuracy_score(prediction, actual)
    scores.append(score)
    return scores, coef

def sgd_logistic(dataset, alpha, epochs):
    b = [0] * len(dataset[0])
    for i in range(epochs):
        print(i)
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
        y_pred = prediction(row, coef)
        predictions.append(round(y_pred))
    return predictions, coef

alpha = 0.01
epochs = 10
scores,coef = evaluate_algorithm(dataset, logisticRegression, alpha, epochs)
print('Scores',scores)
print('Weights',coef)
