import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def KNN_eu(X_train, y_train, X_tests, k):
    predictions = []
    for X_test in X_tests:
        distances = []
        for x_train in X_train:
            distance = euclidean_distance(x_train, X_test)
            distances.append(distance)
        
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    
    return predictions

#
def dM(A,X_i,X_j):
    return np.linalg.norm(np.dot(A, X_i) - np.dot(A, X_i))

def calculate_pij(X, y, A, Omega, i, j):
    numerator = np.exp(-dM(A,X[i],X[j]) ** 2)
    denominator = sum(np.exp(-dM(A,X[i],X[k]) ** 2) for k in Omega[y[i]] if k != i)
    return numerator / denominator if j != i else 0

def objective_function(X, y, A, Omega):
    N = X.shape[0]
    fA = 0
    for i in range(N):
        for j in Omega[y[i]]:
            pij = calculate_pij(X, y, A, Omega, i, j)
            fA += pij
    return fA

def calculate_A(X, y, learning_rate=0.01, max_epoch=100, tol=1e-6):
    N, e = X.shape
    A = np.random.randn(e, e)  
    Omega = []
    for label in range(len(np.unique(y))):
        a = [i for i in range(len(y)) if y[i]==label]
        Omega.append(a)

    pre_fA = objective_function(X, y, A, Omega)
    
    for iteration in range(max_epoch):
        grad = np.zeros_like(A)
        for i in range(N):
            for j in Omega[y[i]]:
                pij = calculate_pij(X, y, A, Omega, i, j)
                grad += -2 * pij * np.outer(X[i] - X[j], X[i] - X[j])
        
        A -= learning_rate * grad
        fA = objective_function(X, y, A, Omega)
        
        if np.abs(fA - pre_fA) < tol:
            break
        pre_fA = fA
    
    return A

def KNN_ma(X_train, y_train, X_tests, A, k):
    predictions = []
    M = np.dot(A, A.T)
    for X_test in X_tests:
        distances = []
        for x_train in X_train:
            delta = X_test - x_train
            distance = np.sqrt(np.dot(np.dot(delta, M), delta.T))
            distances.append(distance)
        
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    
    return predictions

def Accuracy(y_true, y_pred):
    correct_pred = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    total_pred = len(y_true)
    accuracy = correct_pred / total_pred
    return accuracy

