import mindspore.numpy as mnp
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import numpy as np
import struct

context.set_context(mode=context.PYNATIVE_MODE)

class CustomGaussianNB:
    def __init__(self):
        # 初始化模型参数
        self.means = None
        self.variances = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # 获取样本数和特征数
        self.classes = mnp.unique(y) # 获取所有类标签
        n_classes = len(self.classes.asnumpy())  # 类别数

        # 初始化均值、方差和先验概率矩阵
        self.means = mnp.zeros((n_classes, n_features), dtype=mnp.float32)
        self.variances = mnp.zeros((n_classes, n_features), dtype=mnp.float32)
        self.priors = mnp.zeros(n_classes, dtype=mnp.float32)

        for idx, c in enumerate(self.classes.asnumpy()):  # Convert to numpy array for enumeration
            # 获取属于类c的所有样本
            X_c = []
            for i in range(len(y)):
                if(y[i].asnumpy()==c):
                    X_c.append(X[i])
            X_c = Tensor(X_c,dtype=mstype.float32) 
            self.means[idx, :] = mnp.mean(X_c, axis=0) # 计算均值
            self.variances[idx, :] = mnp.var(X_c, axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(n_samples) # 计算先验概率

    def predict(self, X):
        y_pred = [self._predict(x) for x in X] # 对每个样本进行预测
        return mnp.array(y_pred)

    def _predict(self, x):
        posteriors = [] # 后验概率列表

        for idx, c in enumerate(self.classes.asnumpy()):
            prior = mnp.log(self.priors[idx]) # 计算先验概率的对数
            class_conditional = mnp.sum(mnp.log(self._pdf(idx, x))) # 计算类条件概率的对数
            posterior = prior + class_conditional  # 计算后验概率
            posteriors.append(posterior)

        return self.classes[mnp.argmax(posteriors)] # 返回后验概率最大的类标签

    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        numerator = mnp.exp(- (x - mean) ** 2 / (2 * var))
        denominator = mnp.sqrt(2 * mnp.pi * var)
        return numerator / denominator # 返回概率密度函数值

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def Accuracy(y_true, y_pred):
    # 将预测结果转化为同一形式（例如，通过取最大概率的索引）
    if y_pred.shape != y_true.shape:
        y_pred = P.Argmax(axis=1)(y_pred)
    # 计算正确预测的数量
    correct_pred = F.reduce_sum(F.cast(y_true == y_pred, mnp.float32))
    # 总预测数量
    total_pred = y_true.shape[0]
    # 计算精确度
    accuracy = correct_pred / total_pred
    return accuracy
