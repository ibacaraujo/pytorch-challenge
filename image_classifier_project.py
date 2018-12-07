# Developing an AI application

import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

## Load the data

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224,
                                          0.225))])
valid_data_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229,
                                               0.224, 0.225))])

train_images_dataset = datasets.ImageFolder(train_dir, transform=train_data_transform)
valid_images_dataset = datasets.ImageFolder(valid_dir, transform=valid_data_transform)

train_dataloader = torch.utils.data.DataLoader(train_images_dataset, batch_size=32, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_images_dataset, batch_size=32, shuffle=True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Building and training the classifier

vgg16 = models.vgg16(pretrained=True)

for param in vgg16.parameters():
    param.requires_grad_(False)

vgg16.classifier[6] = nn.Linear(4096,102)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vgg16.parameters(), lr=0.01)

n_epochs = 30

valid_loss_min = np.Inf
