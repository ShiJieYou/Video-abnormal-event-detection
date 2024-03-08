import cv2
import torch
import torch.nn as nn
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms   #
import torch.optim as optim
from sklearn.model_selection import train_test_split

cap = cv2.VideoCapture("文件路径/sucai.avi")
frame_index = 0
vdata = []
vlabel = []
sucess = True
while(sucess):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame,(32, 32))
    vdata.append(frame)
    if frame_index >= 527 and frame_index <= 616 or \
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class Mydataset(Dataset.Dataset):
    def __init__(self, vdata, vlabel, transform = None):
        super(Mydataset, self).__init__()
        self.data = vdata
        self.label = vlabel
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

class Net(nn.Module):
    def __init__(self, ndim, n_class):
        super(Net, self).__init__()
        self.feature_enginnering = nn.Sequential(
            nn.Conv2d(ndim, 6, 5),#6*28*28
            nn.MaxPool2d(2,2),#6*14*14
            nn.Conv2d(6, 16, 5),#16*10*10
            nn.MaxPool2d(2, 2),#16*5*5
        )
        self.classfier = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class),
        )

    def forward(self, x):
        out = self.feature_enginnering(x)
        out = out.view(out.size(0), -1)
        out = self.classfier(out)
        return out


if __name__ == "__main__":
    train_data, test_data, train_target, test_target = train_test_split(vdata, vlabel, test_size=0.2)
    train_datasets = Mydataset(train_data, train_target, transform)
    test_datasets = Mydataset(test_data, test_target, transform)
    trainloader = Dataloader.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = Dataloader.DataLoader(test_datasets, batch_size=64, shuffle=True)
    model=Net(3,2)
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
    print(100*correct/len(trainloader))
    print(100 * correct / total)