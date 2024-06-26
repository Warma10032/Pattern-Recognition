import numpy as np
import struct


class CustomGaussianNB:
    def __init__(self):
        # 初始化模型参数
        self.means = None
        self.variances = None
        self.priors = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape  # 获取样本数和特征数
        self.classes = np.unique(y)  # 获取所有类标签
        n_classes = len(self.classes)  # 类别数
        
        # 初始化均值、方差和先验概率矩阵
        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  # 获取属于类c的所有样本
            self.means[idx, :] = X_c.mean(axis=0)  # 计算均值
            self.variances[idx, :] = X_c.var(axis=0) + 1e-9  # 计算方差并添加一个小数以避免除以零
            self.priors[idx] = X_c.shape[0] / float(n_samples)  # 计算先验概率
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]  # 对每个样本进行预测
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []  # 后验概率列表
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])  # 计算先验概率的对数
            class_conditional = np.sum(np.log(self._pdf(idx, x)))  # 计算类条件概率的对数
            posterior = prior + class_conditional  # 计算后验概率
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]  # 返回后验概率最大的类标签
    
    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))  # 计算分子
        denominator = np.sqrt(2 * np.pi * var)  # 计算分母
        return numerator / denominator  # 返回概率密度函数值
    
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
    correct_pred = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    total_pred = len(y_true)
    accuracy = correct_pred / total_pred
    return accuracy