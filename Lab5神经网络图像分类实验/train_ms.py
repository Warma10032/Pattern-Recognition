from mindspore import nn, ops
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore import dtype as mstype
import matplotlib.pyplot as plt

# 继承nn.Cell，类似pytorch.nn.model
class FullyConnectedNN(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size, lr=1e-2):
        # 模块化搭建神经网络
        super(FullyConnectedNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden_size, output_size)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.optimizer = nn.SGD(self.trainable_params(), learning_rate=lr)

    # 完成construct函数，类似forward函数，用于前向传播
    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    # ops.value_and_grad中的优化函数，，mindspore可以在此自动进行梯度下降来降低损失
    def optimization_function(self, data, label):
        output = self(data)
        loss = self.loss_fn(output, label)
        return loss, output

# 实例化网络
network = FullyConnectedNN(3072, 512, 10, lr=0.01)

# 读取数据
cifar10_dataset_dir = "./data/cifar10"

# 使用Cifar10Dataset读取数据并进行数据预处理
train_dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir,usage='train',shuffle=True)
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), 'image')
train_dataset = train_dataset.batch(batch_size=64, drop_remainder=True) # 批处理

test_dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir,usage='test')
test_dataset = test_dataset.map(vision.Rescale(1.0 / 255.0, 0), 'image')
test_dataset = test_dataset.batch(batch_size=64)

# 训练和评估网络
epochs=50
history = []
for epoch in range(epochs):
    network.set_train() # 设置为训练模式
    grad_fn = ops.value_and_grad(network.optimization_function, None, network.optimizer.parameters, has_aux=True)
    total_loss = 0
    for images, labels in train_dataset.create_tuple_iterator(): # 产生迭代器，可自动进行批加载
        labels = labels.astype(mstype.int32)
        (loss, _), grads = grad_fn(images, labels)
        total_loss += ops.depend(loss, network.optimizer(grads))
    train_loss = total_loss / train_dataset.get_dataset_size()
    
    network.set_train(False) # 调整为评估模式
    total, correct = 0, 0
    for images, labels in test_dataset.create_tuple_iterator():
        labels = labels.astype(mstype.int32)
        output = network(images)
        predicted = output.argmax(axis=1)
        correct += (predicted == labels).asnumpy().sum()
        total += len(images)
    accuracy = correct / total
    print(f"Epoch {epoch + 1}, Loss: {train_loss}, Test Accuracy: {accuracy}")
    history.append([epoch + 1, train_loss ,accuracy])

epochs = [x[0] for x in history]
train_losses = [x[1] for x in history]
test_accuracies = [x[2] for x in history]

# 创建图表
plt.figure(figsize=(10, 5))

# 绘制训练损失
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.grid(True)
plt.legend()

# 绘制测试精度
plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.grid(True)
plt.legend()

# 展示图表
plt.show()