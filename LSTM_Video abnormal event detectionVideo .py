import torch

import cv2
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# torch.cuda.is_available() 函数用于检查是否有可用的 GPU。如果有可用的 GPU，就将设备设置为cuda，否则设置为 cpu。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 28
input_size = 32*3
hidden_size = 128
num_layers = 2    # Number of recurrent layers，将两个LSTM堆叠在一起，形成一个“堆叠LSTM”
num_classes = 2
batch_size = 100  #每批次样本的数量
num_epochs = 10
learning_rate = 0.01

# 用于捕获视频文件或硬件设备（如摄像头）的视频流。cap 相当于文件的指针
cap = cv2.VideoCapture("sucai.avi")
frame_index = 0
vdata = []
vlabel = []
sucess = True

while(sucess):
    ret, frame = cap.read()
    if not ret:
        break
    # 因为【神经网络模型通常需要固定大小的输入】，所以我们需要将所有的图像调整到相同的大小。
    frame = cv2.resize(frame, (32, 32))
    vdata.append(frame)
    # 在训练神经网络时，我们需要这些标签作为真实的目标值（ground truth）来计算损失（loss）。神经网络的目标就是尽可能地减小预测值与真实标签之间的差距。
    if (frame_index >= 527 and frame_index <= 616
            or frame_index >= 1332 and frame_index <= 1441
            or frame_index >= 1808 and frame_index <= 1987
            or frame_index >= 2607 and frame_index <= 2686
            or frame_index >= 3221 and frame_index <= 3430
            or frame_index >= 3940 and frame_index <= 4019
            or frame_index >= 4809 and frame_index <= 4930
            or frame_index >= 5424 and frame_index <= 5597
            or frame_index >= 6197 and frame_index <= 6236
            or frame_index >= 6885 and frame_index <= 6914
            or frame_index >= 7702 and frame_index <= 7739):
        vlabel.append(1)
    else:
        vlabel.append(0)
    frame_index = frame_index + 1
# print(frame_index)    #7739
# print(len(vdata), *np.array(vdata).shape[1:])  #7739 32 32 3
# print(len(vlabel))                             #7739


# cap.release()和cv2.destroyAllWindows()是用来 停止捕获视频 和 关闭相应的显示窗口的 。用来停止捕获视频和关闭相应的显示窗口的,在操作完成之后需要释放，否则其他程序无法再次获取摄像头或者视频文件。
cap.release()
cv2.destroyAllWindows()

# 得到的张量是一个四维的张量，具体形状取决于原始图像的尺寸和通道数。
# 张量的形状通常使用(batch_size, channels, height, width)的顺序表示,batch_size 表示张量中的样本数量。
transformer = transforms.Compose([
                       # 将输入的图像数据转换为张量（tensor）类型。这个操作将【图像从PILImage或者NumPy ndarray转换为Tensor类型】，同时【对图像的像素值做标准化（从0-255范围转换为0-1范围）】。并且把【（H、W，C）-->（C、H、W）】
                       transforms.ToTensor(),
                       # 是一个归一化操作，用于对图像进行标准化处理,【Normalize】操作会将每个通道的数据减去对应的均值，然后除以对应的标准差，【将数据的分布调整为标准正态分布】。会将每个通道的数据减去对应的均值，然后除以对应的标准差。均值和标准差都是以R、G、B通道顺序为参数传入的元组
                       transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                            std=(0.5, 0.5, 0.5))
                                ])

class dataset(Dataset):
    #  __init__方法是类的初始化方法，当创建类的实例（对象）时，会自动调用这个方法。这些属性可以在类的其他方法中使用。
    def __init__(self, Data, Label, transform=None):
        self.data = Data
        self.label = Label
        self.transform = transform

    # len() 函数用于获取张量（Tensor）的第一个维度上的大小。它返回的结果是张量的第一个维度的长度。如果张量是一个一维向量，则返回向量中元素的个数。
    def __len__(self):
        return len(self.data)

    # index参数表示要获取的数据在数据集中的索引值。
    def __getitem__(self, index):
        # 获取索引为item的样本的数据和标签
        data = self.data[index]
        label = self.label[index]
        # 判断是否定义了数据变换方法。如果定义了数据变换方法（即self.transform不为None），那么就对数据进行变换
        if self.transform is not None:
            data = self.transform(data)
        return data, label



class CNN_RNN(nn.Module):
    def __init__(self, ndim, input_size, hidden_size, num_layers, num_directions, n_class):
        # 通常情况下，我们在子类中定义了和父类同名的方法，那么子类的方法就会覆盖父类的方法。而super关键字实现了对父类方法的改写(增加了功能，增加的功能写在子类中，父类方法中原来的功能得以保留)。也可以说，super关键字帮助我们实现了在子类中调用父类的方法
        super(CNN_RNN, self).__init__()
        # self.input_
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        # 从输入的多维数据中提取特征。
        # [ (图大小 + 2*填充 - 卷积核大小)/步长 ]+1
        self.feature_enginnering = nn.Sequential(
            nn.Conv2d(ndim, 6, 5),           # [(32+2*0-5)/1]+1 = 28    --> 6*28*28 (第一层卷积完成后生成特征图大小)
            nn.MaxPool2d(2, 2),                   # (28*28)/2*2 = 14*14      --> 6*14*14 (池化后特征图大小)
            nn.Conv2d(6, 16, 5),  # [(14+2*0-5)/1]+1 = 10    --> 16*10*10 (第二层卷积完成后生成特征图大小)
            nn.MaxPool2d(2, 2)                    # (10*10)/2*2 = 5*5        --> 16*5*5  (池化后特征图大小)
        )
        # 当batch_first=True时，输出的形状为[batch_size, seq_len, hidden_size]；当batch_first=False（默认值）时，形状为[seq_len, batch_size, hidden_size]。
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, n_class)

    # x- 64 3 32 32
    def forward(self, x):
        # 初始化LSTM的初始[隐藏状态h0]和[细胞状态c0]，初始化为全零【张量】
        # 4 64 128
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)

        out = self.feature_enginnering(x)    # 64 16 5 5

        # 通过这行代码，将特征图out的形状变为(batch_size, num_channels, flattened_size)，其中flattened_size表示将高度和宽度维度展平后的大小。
        # 这样可以将特征图的空间信息转换为一维的特征向量，方便传入后续的LSTM模块进行序列建模。
        out = out.view(out.size(0), out.size(1), -1)     # 64 16 25

        # LSTM的输出形状(batch，seq_len, hidden_size*2)
        # h0,c0 - 4 64 128
        # out: tensor of shape(batch_size, seq_length, hidden_size * 2)
        out, (h0, c0) = self.lstm(out, (h0, c0))   # 输出 out - 64 16 256 (hidden_size*2 = 128*2 = 256),
        # h_n = output[:, -1, :]
        # 表示从output张量中取出所有batch_size的最后一个时间步的输出。
        c = out[:, -1, :]    # 64 256   [batch_size, hidden_size*num_directions]

        out = self.fc(c)     # 64 2
        # out = self.fc(out[:, -1, :])
        return out

if  __name__ == "__main__":

    # vdata[7739 32 32 3], vlabel=7739 ----划分后的训练集数据、测试集数据、训练集标签和测试集标签----> train_data[6191 32 32 3],train_targer[6191],test_data[1548 32 32 3],test_target[1548]
    train_data, test_data, train_target, test_target = train_test_split(vdata, vlabel, test_size=0.2)


    train_dataset = dataset(train_data, train_target, transformer)     # train_dataset.data[6191 3 32 32],train_dataset.label[6191]
    test_dataset = dataset(test_data, test_target, transformer)        # test_dataset.data[1548 3 32 32],test_dataset.label[1548]

    # shuffle=False 表示在每个 epoch（整个数据集被遍历一次）中不对数据进行重新洗牌。 num_workers=0 表示在数据加载过程中使用的子进程数量为0，即不使用多线程。
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 64 2
    model = CNN_RNN(3, 5*5, 128, 2, 2, 2)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(trainloader)   # 97  train_dataset(6191) / batch_size(64) = 97
    for epoch in range(num_epochs):
        # enumerate(trainloader) 是一个迭代器，它将每个批次的图像和标签打包成一个元组（imgs, labels），并给它们赋予一个索引值 i。每次迭代都会从数据加载器中取出一个批次的图像和标签。
        # i 是索引值，从 0 开始递增，用于记录当前是第几个批次。
        for i, (imgs, labels) in enumerate(trainloader):
            # pyTorch 0.4之前，Tensor是不能计算梯度的，所以需要Variable类进行封装，才能构建计算图。但是再PyTorch0.4 以后，合并了Tensor和Vairable类，可直接计算Tensor梯度。
            imgs = Variable(imgs)
            labels = Variable(labels)
            outputs = model(imgs)             # 64 2
            loss = criterion(outputs, labels)
            # 在计算新的梯度之前，先将模型参数的梯度清零。
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


    # torch.no_grad() 声明了一个上下文环境，在这个环境中，PyTorch 将不会计算梯度，从而节省内存并加快计算速度。
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = model(images)
            # 找到模型输出中概率最大的类别作为预测结果。
            # _是一个占位符，用于接收最大值，而pred是一个变量，用于接收最大值对应的索引。这样做是因为在计算准确率时，我们只关心预测的类别索引，而不关心具体的最大值。
            # torch.max(outputs.data, 1) 中，1 是一个参数，用于指定在哪个维度上进行最大值计算。  【1  在维度1 上计算最大值】
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(labels.size(0))
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

        # Save the model checkpoint
    #保存模型的状态字典（state_dict）到文件 'model.ckpt'
    # model.state_dict() 可以获取模型的当前状态字典，其中包含了每个参数的名称和对应的数值。
    torch.save(model.state_dict(), 'model.ckpt')