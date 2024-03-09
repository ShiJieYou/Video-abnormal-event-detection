import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.utils.data.dataset as Dataset
# from torch.utils.data import Dataset,DateLoader
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transform
from sklearn.model_selection import train_test_split


# 所有的框架都在torchversion.models里面
import torchvision.models as models
model_vgg = models.vgg16(pretrained=True)
for param in model_vgg.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture("sucai.avi")
frame_index = 0
vdata = []
vlabel = []
sucess = True

while (sucess):
    ret, frame = cap.read()
    if not ret:
        break
    #将每一帧的图像大小调整为32x32像素。这是因为神经网络模型通常需要固定大小的输入，所以我们需要将所有的图像调整到相同的大小。
    frame = cv2.resize(frame, (224, 224))
    # 打标签
    vdata.append(frame)
    if 527 <= frame_index <= 616 or \
            frame_index >= 1332 and frame_index <= 1441 or \
            frame_index >= 1808 and frame_index <= 1987 or \
            frame_index >= 2607 and frame_index <= 2686 or \
            frame_index >= 3221 and frame_index <= 3430 or \
            frame_index >= 3940 and frame_index <= 4019 or \
            frame_index >= 4809 and frame_index <= 4930 or \
            frame_index >= 5424 and frame_index <= 5597 or \
            frame_index >= 6197 and frame_index <= 6236 or \
            frame_index >= 6885 and frame_index <= 6914 or \
            frame_index >= 7702 and frame_index <= 7739:
        vlabel.append(1)
    else:
        vlabel.append(0)
    frame_index = frame_index + 1
cap.release()
cv2.destroyAllWindows()

transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]

)


class MyDataset(Dataset.Dataset):
    def __init__(self, vdata, vlabel, transform=None):
        super(MyDataset, self).__init__()
        self.data = vdata
        self.label = vlabel
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

class Net(nn.Module):  # nn.Modle 句柄
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.features_extractor = model_vgg.features
        self.vgg.classifier[6] = nn.Linear(4096, n_class) #修改最后一层全连接层
        self.classifier = self.vgg.classifier

    def forward(self, x):
        out = self.features_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    train_data, test_data, train_target, test_target = train_test_split(vdata, vlabel, test_size=0.2)
    train_datasets = MyDataset(train_data, train_target, transform)
    test_datasets = MyDataset(test_data, test_target, transform)
    trainloader = Dataloader.DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
    testloader = Dataloader.DataLoader(test_datasets, batch_size=64, shuffle=True, num_workers=0)

    model = Net(2).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            imgs, labels = data
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 0:
                print("[{},{}] loss={}".format(epoch + 1, i + 1, running_loss / len(trainloader)))

    correct = 0.0
    total = 0
    for data in testloader:
        imgs, label = data
        outputs = model(imgs)
        _, pred = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()

    print(100 * correct / total)
