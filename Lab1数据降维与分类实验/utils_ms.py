import mindspore.numpy as mnp
import numpy as np
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F

context.set_context(mode=context.PYNATIVE_MODE)

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.weight = None
        self.history = []

    def init_weights(self, num_features, num_classes):
        self.weight = np.random.randn(num_features, num_classes)
        self.weight = Tensor(self.weight,dtype=mstype.float32)

    def forward(self, X):
        return X @ self.weight

    def backward(self, X, y_true, y_preds):
        num_samples = X.shape[0]
        grad = X.T @ (y_preds - y_true) / num_samples
        self.weight -= self.lr * grad

    # 训练步骤代码，包含前向传播和反向传播
    def train(self, X, y):
        X = mnp.column_stack([X, mnp.ones((X.shape[0], 1))])
        num_samples, num_features = X.shape
        num_classes = len(mnp.unique(y))
        self.init_weights(num_features, num_classes) # 初始化权重

        y_one_hot = one_hot_encode(y, num_classes)

        for _ in range(self.epochs):
            model = self.forward(X)
            y_preds = softmax(model)
            loss = cross_entropy_loss(y_one_hot, y_preds)
            self.history.append(loss) # 保存训练损失
            self.backward(X, y_one_hot, y_preds)

    # 预测步骤代码，用于评估模型，计算准确度
    def predict(self, X):
        X = mnp.column_stack([X, mnp.ones((X.shape[0], 1))])
        model = self.forward(X)
        y_preds = softmax(model)
        return mnp.argmax(y_preds, axis=1)

def softmax(z):
    exp_z = mnp.exp(z - mnp.max(z, axis=1, keepdims=True))
    return exp_z / mnp.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = mnp.clip(y_pred, epsilon, 1 - epsilon)
    loss = -mnp.sum(y_true * mnp.log(y_pred)) / y_true.shape[0]
    return loss

def one_hot_encode(y, num_classes=2):
    return mnp.eye(num_classes)[Tensor(y, dtype=mnp.int32)]

def PCA(X, n_components):
    # 数据标准化
    X_meaned = X - np.mean(X, axis=0)
    # 计算协方差矩阵
    cov_matrix = np.cov(X_meaned, rowvar=False)
    # 特征值分解
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    # 特征向量排序
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # 选择前n个特征向量（主成分）
    eigenvector_subset = sorted_eigenvectors[:, :n_components]
    # 投影数据
    X_pca = np.dot(X_meaned, eigenvector_subset)
    return X_pca

def LDA(X, y, n_components):
    # 计算类别相同的葡萄酒的均值向量
    class_labels = np.unique(y)
    mean_vectors = []
    for label in class_labels:
        mean_vectors.append(np.mean(X[y == label], axis=0))
    # 计算类内散布矩阵 S_W
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for label, mean_vec in zip(class_labels, mean_vectors):
        class_scatter = np.zeros((X.shape[1], X.shape[1]))
        for row in X[y == label]:
            row, mean_vec = row.reshape(X.shape[1], 1), mean_vec.reshape(X.shape[1], 1)
            class_scatter += (row - mean_vec).dot((row - mean_vec).T)
        S_W += class_scatter
    # 计算类间散布矩阵 S_B
    μ1 = mean_vectors[0].reshape(X.shape[1], 1)
    μ2 = mean_vectors[1].reshape(X.shape[1], 1)
    S_B = np.dot((μ1-μ2),(μ1-μ2).T)
    # 使用 SVD 计算特征值
    inv_S_W = np.linalg.pinv(S_W)
    mat = inv_S_W.dot(S_B)
    U, s, Vh = np.linalg.svd(mat)
    # 选择前n个特征向量
    W = U[:, :n_components]
    # 投影数据
    X_lda = np.dot(X, W)
    return X_lda

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    # 通过打乱索引来实现随机抽取
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    train_size = n_samples - test_size
    
    X_train = X[indices[0:train_size]]
    X_test = X[indices[train_size:]]
    y_train = y[indices[0:train_size]]
    y_test = y[indices[train_size:]]
    
    return X_train, X_test, y_train, y_test

def Accuracy(y_true, y_pred):
    # 将预测结果转化为同一形式
    if y_pred.shape != y_true.shape:
        y_pred = P.Argmax(axis=1)(y_pred)
    # 计算正确预测的数量
    correct_pred = F.reduce_sum(F.cast(y_true == y_pred, mnp.float32))
    # 总预测数量
    total_pred = y_true.shape[0]
    # 计算精确度
    accuracy = correct_pred / total_pred
    return accuracy
