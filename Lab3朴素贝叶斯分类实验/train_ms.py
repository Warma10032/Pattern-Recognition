from utils_ms import *
import matplotlib.pyplot as plt
from mindspore import Tensor

# 指定数据文件的路径
train_images_path = './data/train-images.idx3-ubyte'
train_labels_path = './data/train-labels.idx1-ubyte'
test_images_path = './data/t10k-images.idx3-ubyte'
test_labels_path = './data/t10k-labels.idx1-ubyte'

# 读取数据
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

train_images_Tensor = Tensor(train_images,dtype=mstype.float32)
train_labels_Tensor = Tensor(train_labels,dtype=mstype.float32)
test_images_Tensor = Tensor(test_images,dtype=mstype.float32)
test_labels_Tensor = Tensor(test_labels,dtype=mstype.float32)

# 初始化自定义朴素贝叶斯分类器
custom_gnb = CustomGaussianNB()

# 训练模型
custom_gnb.fit(train_images_Tensor, train_labels_Tensor)

# 预测测试集
y_pred = custom_gnb.predict(test_images_Tensor)

# 计算准确率
accuracy = Accuracy(test_labels_Tensor, y_pred)
accuracy_scalar = accuracy.asnumpy().item()
print(f"Accuracy: {accuracy_scalar * 100:.2f}%")

# 可视化部分测试样本及其预测结果
def plot_samples(X, y, y_pred, n_samples=10):
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}\n Pred: {y_pred[i]}")
        plt.axis('off')
    plt.show()

# 展示前10个测试样本及其预测结果
plot_samples(test_images, test_labels, y_pred, n_samples=10)