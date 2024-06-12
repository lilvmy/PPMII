import logging as log

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import *
from torch.utils.data import DataLoader, random_split
from medmnist import PneumoniaMNIST


def setup_logging():
    root_logger = log.getLogger()
    root_logger.setLevel(log.INFO)
    handler = log.FileHandler('ModelConstruction.log', 'w', 'utf-8')
    handler.setFormatter(log.Formatter(fmt='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    root_logger.addHandler(handler)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([32,32]),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = PneumoniaMNIST(split="train", download=True, size=28, transform=data_transforms,
                           root="/home/cysren/Desktop/ld_file/PPMII/PneumoniaMNIST")

test_dataset = PneumoniaMNIST(split="test", download=True, size=28, transform=data_transforms,
                           root="/home/cysren/Desktop/ld_file/PPMII/PneumoniaMNIST")




train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64)

class taylor_exapnsion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.1992 + torch.multiply(x, 0.5002) + torch.multiply(torch.pow(x, 2), 0.1997)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train_net(network, epochs, device):
    optimizer = optim.Adam(network.parameters(), lr = 0.001)
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch

            # the labels of PneumoniaMNIST are 2-dimension not 1-dimension
            # 2-dimension (tesnor([[2],[3],......])) 1-dimension (tensor([1,2,3,...]))
            labels = labels.reshape(-1)
            images, labels = images.to(device), labels.to(device)

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct +=get_num_correct(preds, labels)


def test_net(network, device):
    network.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            labels = labels.reshape(-1)
            images, labels = images.to(device), labels.to(device)

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        accuracy = round(100. * (total_correct / len(test_loader.dataset)), 4)

    return total_correct / len(test_loader.dataset)


# LeNet-5 model add 2 BN, 2 ReLU
class PneumoniaMNIST(nn.Module):
    def __init__(self):
        super(PneumoniaMNIST, self).__init__()
        self.pneumoniaMNIST = nn.Sequential(
            Conv2d(1, 6, kernel_size=5),
            BatchNorm2d(6),
            ReLU(),
            AvgPool2d(kernel_size=2,stride=2),

            Conv2d(6, 16, kernel_size=5),
            AvgPool2d(kernel_size=2, stride=2),
            # BatchNorm2d(16),
            ReLU(),


            Flatten(),
            Linear(16*5*5, 120),
            Linear(120, 84),
            Linear(84, 2)
        )

    def forward(self, x):
        x = self.pneumoniaMNIST(x)

        return x

# LeNet-5 model add 2 BN, 2 taylor polynomial
class PneumoniaMNIST_2taylor(nn.Module):
    def __init__(self):
        super(PneumoniaMNIST_2taylor, self).__init__()
        self.pneumoniaMNIST_2taylor = nn.Sequential(
            Conv2d(1, 6, kernel_size=5),
            BatchNorm2d(6),
            taylor_exapnsion(),
            AvgPool2d(kernel_size=2,stride=2),

            Conv2d(6, 16, kernel_size=5),
            AvgPool2d(kernel_size=2, stride=2),
            # BatchNorm2d(16),
            taylor_exapnsion(),


            Flatten(),
            Linear(16*5*5, 120),
            Linear(120, 84),
            Linear(84, 2)
        )

    def forward(self, x):
        x = self.pneumoniaMNIST_2taylor(x)

        return x



if __name__ == "__main__":
    setup_logging()
    experiment = 10
    accuracies1 = []
    accuracies2 = []

    for i in range(0, experiment):
        net = PneumoniaMNIST().to(device)
        train_net(net, 36, device)
        acc = test_net(net, device)
        accuracies1.append(acc)

    m = np.array(accuracies1)
    log.info(f"Results for PneumoniaMNIST model_2ReLU:")
    log.info(f"the 10 accuracy of 35 epoch are: {m}")
    log.info(f"Mean accuracy on test_set of 35 epoch: {np.mean(m)}")
    log.info(f"Var of  epoch: {np.var(m)}")
    torch.save(net.state_dict(), f"PneumoniaMNIST_epoch_35_2ReLU.pth")


    for j in range(0, experiment):
        model = PneumoniaMNIST_2taylor()
        model.to(device)
        train_net(model, 35, device)
        acc = test_net(model, device)
        accuracies2.append(acc)


    m = np.array(accuracies2)
    log.info(f"Results for PneumoniaMNIST model 2-degree Taylor Expansion:")
    log.info(f"the 10 accuracy of 35 epoch are: {m}")
    log.info(f"Mean accuracy on test_set of 35 epoch: {np.mean(m)}")
    log.info(f"Var of  epoch: {np.var(m)}")
    torch.save(model.state_dict(), f"PneumoniaMNIST_epoch_35_2taylor.pth")

