import numpy as np
import pickle
import math

def relu(x):
    return np.maximum(0, x)

# relu函数的导数
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def one_hot_encode(labels, num_classes=10):
    # 创建一个形状为 (len(labels), num_classes) 的零矩阵
    one_hot_labels = np.zeros((len(labels), num_classes))
    # 设置对应的索引为1
    one_hot_labels[np.arange(len(labels)), labels] = 1  
    return one_hot_labels

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.sum(y_true * np.log(p + 1e-12))
    loss = log_likelihood / m
    return loss

class FullyConnectedNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def __call__(self, inputs):
        return self.forward(inputs)

    # 前向传播
    def forward(self, x):
        # 输入层到隐藏层
        self.z1 = x @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        # 隐藏层到输出层
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    # 反向传播算法
    def backward(self, x, y):
        # 在反向传播前进行了一次前向传播，利用其中的数据可以简化梯度计算
        z1 = x @ self.w1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = softmax(z2)

        # 反向传播到输出层
        batch_size = x.shape[0]
        delta2 = a2 - y
        grad_w2 = a1.T @ delta2 / batch_size
        grad_b2 = np.sum(delta2, axis=0) / batch_size

        # 反向传播到隐藏层
        delta1 = (delta2 @ self.w2.T) * relu_derivative(z1)
        grad_w1 = x.T @ delta1 / batch_size
        grad_b1 = np.sum(delta1, axis=0) / batch_size

        # 更新参数
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

                
class CIFAR10Dataset:
    def __init__(self, file_paths):
        self.data, self.labels = self.load_data(file_paths)

    def load_data(self, file_paths):
        data = []
        labels = []
        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
                labels.extend(batch[b'labels'])
                images = batch[b'data']
                images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1) # WHC2CWH
                data.extend(images)
        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 返回第index个图像及其标签
        return self.data[index], self.labels[index]
    
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    # 创造迭代器
    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[idx] for idx in indices]
        data_batch = np.array([item[0] for item in batch])
        labels_batch = np.array([item[1] for item in batch])
        labels_batch = one_hot_encode(labels_batch)  
        self.current_idx += self.batch_size
        return data_batch, labels_batch
    
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

