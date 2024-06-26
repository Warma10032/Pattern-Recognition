import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mindspore.numpy as np
from mindspore import Tensor
from utils_ms import *

# 读取CSV文件
df_red = pd.read_csv(r'data/winequality-red.csv', sep=';')
df_white = pd.read_csv(r'data/winequality-white.csv', sep=';')

df_red['label'] = 0
df_white['label'] = 1

df = pd.concat([df_red, df_white], axis=0, ignore_index=True)

# 提取特征和标签
X = df.drop('label', axis=1).values
y = df['label'].values

X_pca = PCA(X,n_components=2)
X_lda = LDA(X,y,n_components=2)

# 可视化数据
feature1 = 0 # "volatile acidity"
feature2 = 2 # "residual sugar"
# 可视化原始数据（指定两个属性）
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
for label in np.unique(y):
    plt.scatter(X[y == label, feature1], X[y == label, feature2], label=f'Class {label}')
plt.title('Original Data (Two Attributes)')
plt.xlabel(f'volatile acidity')
plt.ylabel(f'residual sugar')
plt.legend()

# 可视化PCA降维到二维的数据
plt.subplot(1, 3, 2)
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f'Class {label}')
plt.title('PCA Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 可视化LDA降维到二维的数据
plt.subplot(1, 3, 3)
for label in np.unique(y):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=f'Class {label}')
plt.title('LDA Reduced Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()

plt.tight_layout()
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.2, random_state=42)
X_train = Tensor(X_train,dtype=mstype.float32)
X_test = Tensor(X_test,dtype=mstype.float32)
y_train = Tensor(y_train,dtype=mstype.float32)
X_train = Tensor(X_train,dtype=mstype.float32)
y_test = Tensor(y_test,dtype=mstype.float32)
X_train_pca = Tensor(X_train_pca,dtype=mstype.float32)
X_test_pca = Tensor(X_test_pca,dtype=mstype.float32)
y_train_pca = Tensor(y_train_pca,dtype=mstype.float32)
y_test_pca = Tensor(y_test_pca,dtype=mstype.float32)
X_train_lda = Tensor(X_train_lda,dtype=mstype.float32)
X_test_lda = Tensor(X_test_lda,dtype=mstype.float32)
y_train_lda = Tensor(y_train_lda,dtype=mstype.float32)
y_test_lda = Tensor(y_test_lda,dtype=mstype.float32)

classifier = LogisticRegression(lr=0.001,epochs=10000)
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = Accuracy(y_test, y_pred)
loss1 = classifier.history

classifier_pca = LogisticRegression(lr=0.001,epochs=10000)
classifier_pca.train(X_train_pca, y_train_pca)
y_pred_pca = classifier_pca.predict(X_test_pca)
accuracy_pca = Accuracy(y_test_pca, y_pred_pca)
loss2 = classifier_pca.history

classifier_lda = LogisticRegression(lr=0.001,epochs=10000)
classifier_lda.train(X_train_lda, y_train_lda)
y_pred_lda = classifier_lda.predict(X_test_lda)
accuracy_lda = Accuracy(y_test_lda, y_pred_lda)
loss3 = classifier_lda.history

# 绘制损失变化图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  
rcParams['font.sans-serif'] = ['SimHei']  

axs[0].plot(loss1, marker='o', linestyle='-', color='blue')
axs[0].set_title('直接分类损失')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].plot(loss2, marker='o', linestyle='-', color='red')
axs[1].set_title('PCA降维后分类损失')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')

axs[2].plot(loss3, marker='o', linestyle='-', color='green')
axs[2].set_title('LDA降维后分类损失')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')

plt.tight_layout()

plt.show()

print(f'直接分类准确率: {accuracy}')
print(f'PCA降维后的分类准确率: {accuracy_pca}')
print(f'LDA降维后的分类准确率: {accuracy_lda}')