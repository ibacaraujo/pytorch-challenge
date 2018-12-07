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

print('Beginning training...')
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    valid_loss = 0.0
    # train the model
    vgg16.train()
    print('Training...')
    for data, target in train_dataloader:
        print('For data, target...')
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = vgg16(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    # validate the model
    vgg16.eval()
    print('Validation...')
    for data, target in valid_dataloader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_dataloader.dataset)
    valid_loss = valid_loss/len(valid_dataloader.dataset)
    
    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(
            valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model_flowers.pt')
        valid_loss_min = valid_loss
