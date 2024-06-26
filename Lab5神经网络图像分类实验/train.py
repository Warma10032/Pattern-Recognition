from utils import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 加载数据
train_files = [f'data/cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)]
test_file = ['data/cifar-10-batches-py/test_batch']

train_dataset = CIFAR10Dataset(train_files)
test_dataset = CIFAR10Dataset(test_file)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 实例化网络
network = FullyConnectedNN(3072, 512, 10)  

epochs = 50

# 训练循环
history = [] # 用于记录评估参数
for epoch in range(epochs):
    total_loss = 0
    # 批处理，批大小为512
    for data, label in train_dataloader:
        predictions = network.forward(data.reshape(data.shape[0], -1))
        loss = cross_entropy_loss(predictions, label)
        network.backward(data.reshape(data.shape[0], -1), label)
        total_loss += loss
    trian_loss = total_loss / len(train_dataloader)    
    print(f'Epoch {epoch + 1}, Train Loss: {trian_loss}')
    
    # 评估模型
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for data, labels in test_dataloader:
        predictions = network.forward(data.reshape(data.shape[0], -1))
        loss = cross_entropy_loss(predictions, labels)
        total_loss += loss
        predicted_labels = np.argmax(predictions, axis=1) # 选择最大概率值对应的标签作为预测
        labels = np.argmax(labels, axis=1)
        correct_predictions += np.sum(predicted_labels == labels)
        total_samples += len(labels)
    test_loss = total_loss / len(test_dataloader) # 测试损失
    test_accuracy = correct_predictions / total_samples # 测试精度
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    
    history.append([epoch + 1,trian_loss,test_loss,test_accuracy])
    
epochs = [x[0] for x in history]
train_losses = [x[1] for x in history]
test_losses = [x[2] for x in history]
test_accuracies = [x[3] for x in history]

# 创建图表
plt.figure(figsize=(10, 5))

# 绘制训练损失和测试损失
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
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
    
