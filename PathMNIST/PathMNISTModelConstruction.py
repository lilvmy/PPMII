import logging as log
import torch
import torchvision.transforms as transforms
from medmnist import PathMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import *
import numpy as np


def setup_logging():
    root_logger = log.getLogger()
    root_logger.setLevel(log.INFO)
    handler = log.FileHandler("PathMNISTModelConstruction.log", "w", "utf-8")
    handler.setFormatter(log.Formatter(fmt="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(handler)

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([32,32]),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = PathMNIST(split="train", download=True, transform=data_transforms,
                          size = 28, root="/home/cysren/Desktop/ld_file/PPMII/PathMNIST")
test_dataset = PathMNIST(split="test", download=True, transform=data_transforms,
                         size = 28, root="/home/cysren/Desktop/ld_file/PPMII/PathMNIST")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64)

class taylor_expansion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.1992 + torch.multiply(x, 0.5002) + torch.multiply(torch.pow(x, 2), 0.1997)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train_net(network, epochs, device):
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch
            labels = labels.reshape(-1)
            images, labels = images.to(device), labels.to(device)

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

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

    return total_correct /len(test_loader.dataset)

class PathMNIST(nn.Module):
    def __init__(self):
        super(PathMNIST, self).__init__()
        self.pathMNIST = nn.Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2),
            BatchNorm2d(64),
            ReLU(),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            BatchNorm2d(128),
            ReLU(),

            Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            AvgPool2d(kernel_size=2),
            ReLU(),

            Flatten(),
            Linear(2*2*256, 128),
            Linear(128, 9)
        )

    def forward(self, x):
        x = self.pathMNIST(x)

        return x

class PathMNIST_2taylor(nn.Module):
    def __init__(self):
        super(PathMNIST_2taylor, self).__init__()
        self.pathMNIST_2taylor = nn.Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2),
            BatchNorm2d(64),
            taylor_expansion(),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            BatchNorm2d(128),
            taylor_expansion(),

            Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            AvgPool2d(kernel_size=2),
            taylor_expansion(),

            Flatten(),
            Linear(2 * 2 * 256, 128),
            Linear(128, 9)
        )

    def forward(self, x):
        x = self.pathMNIST_2taylor(x)

        return x

if __name__ == "__main__":
    setup_logging()
    experiment = 10
    accuracies1 = []
    accuracies2 = []

    for i in range(0, experiment):
        net = PathMNIST().to(device)
        train_net(net, 35, device)
        acc = test_net(net, device)
        accuracies1.append(acc)

    m = np.array(accuracies1)
    log.info(f"Results for PathMNIST:")
    log.info(f"the 10 accuracy of 35 epoch are: {m}")
    log.info(f"Mean accuracy on test_dataset of 35 epoch: {np.mean(m)}")
    log.info(f"Var of 35 epoch: {np.var(m)}")
    torch.save(net.state_dict(), f"PathMNIST_epoch_35_2ReLU")


    for j in range(0, experiment):
        net = PathMNIST_2taylor().to(device)
        train_net(net, 35, device)
        acc = test_net(net, device)
        accuracies2.append(acc)

    m = np.array(accuracies2)
    log.info(f"Results for PathMNIST_2taylor:")
    log.info(f"the 10 accuracy of 35 epoch are: {m}")
    log.info(f"Mean accuracy on test_dataset of 35 epoch: {np.mean(m)}")
    log.info(f"Var of 35 epoch: {np.var(m)}")
    torch.save(net.state_dict(), f"PathMNIST_epoch_35_2taylor")


