import numpy as np
import pandas as pd
from collections import Counter
from utils import *

train_data = pd.read_csv('./data/data/train.csv')
val_data = pd.read_csv('./data/data/val.csv')
test_data = pd.read_csv('./data/data/test_data.csv')

X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

X_val = val_data.drop('label', axis=1).values
y_val = val_data['label'].values

X_test = test_data.values

accuracy = []
y_test_pred = []
for k in range(1,10):
    y_val_pred = KNN_eu(X_train,y_train,X_val,k)
    accuracy.append(Accuracy(y_val,y_val_pred))
    y_test_pred.append(KNN_eu(X_train,y_train,X_test,k))

print(accuracy)
k_best = np.argmax(accuracy)+1
print(f'在验证集上分类准确度最优的分类器对测试集的分类结果为{y_test_pred[k_best]}')

A = calculate_A(X_train,y_train)
y_val_pred = KNN_ma(X_train,y_train,X_val,A,3)
print(Accuracy(y_val,y_val_pred))
