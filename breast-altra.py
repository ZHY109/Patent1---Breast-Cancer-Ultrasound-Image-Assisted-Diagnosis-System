#!/usr/bin/env python
# coding: utf-8

# Here, I set to mark areas of the breast ultrasound scans that are cancerous using image segmentation with deep learning. To this goal, I designed a UNet architecture from scratch (in pytorch) which yielded a strong performance measured by Dice index. 
# 
# **Dice index was around 70% when averaged over test data samples.**
# <br><br>
# The analysis steps:
# * Import libraries
# * Collect data path for images
# * Load a subset of data to estimate mean and std for normalization purposes
# * Use the mean and std to create image transforms/augmentations
# * Separately load data samples into train and test sets
# * Visual inspection of data
# * Design the UNet
# * Instantiate a UNet
# * Train the model
# * Visualize model performance across epochs
# * Save the model
# * Visually inspect some of the segmentations in comparison to the ground truth
# * Check for potential biases in model performance across cancer types
# 

# # Import libraries

# In[69]:


import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from PIL import Image

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision.io import read_image

import re

data_path = 'E:/datasets/Dataset_BUSI_with_GT'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# # Collect image paths

# In[70]:


image_list = []

folders = os.listdir(data_path)
for folder in folders:
    if folder != 'normal':
        files = os.listdir( os.path.join(data_path, folder) )
        for file in files:
            if 'mask' not in file:
                image_list.append(data_path+'/'+folder+'/'+file) 

random.shuffle(image_list)

data_length = len( image_list )

image_list_train = image_list[:np.floor(0.9*data_length).astype(int)]
image_list_test = image_list[np.floor(0.9*data_length).astype(int):]

print( f'Total length of data: {data_length}')


# # define hyperparameters

# In[71]:


BATCH_SIZE = 16
IMAGE_SIZE = 64
CHANNEL_LIST = [1, 16, 32, 64]

N_EPOCHS = 100

class_to_inx = {'benign': 0, 'malignant': 1}
inx_to_class = {0: 'benign', 1: 'malignant'}


# # Define Data class

# In[78]:


class BreastData(Dataset):
    def __init__(self, image_list, data_mode='Train', transform=None):
        self.image_list = image_list
        self.data_mode = data_mode
        self.image_transform = transform[0]
        self.mask_transform = transform[1]
        
    def __len__(self):
        return len(self.image_list)
    
    def extract_cancer_type(self, image_path):
        if 'benign' in image_path:
            cancer_type = 0
        elif 'malignant' in image_path:
            cancer_type = 1
        return cancer_type
    
    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        
        mask_path = image_path.split('.')[0] + '_mask.png'
        mask = Image.open(mask_path).convert('1')
        
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
            
        cancer_type = self.extract_cancer_type(image_path)
        return image, mask, cancer_type


# # Define Tversky loss as a performance metric

# In[79]:


class tverskyloss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, pred, target):
        loss = 1 - tversky_index(pred, target, alpha=self.alpha, beta=self.beta, reduction=self.reduction)
        return loss

def tversky_index(y_pred, y_true, alpha=0.5, beta=0.5, reduction='mean'):
    # generalization of dice coefficient algorithm
    #   alpha corresponds to emphasis on False Positives
    #   beta corresponds to emphasis on False Negatives (our focus)
    #   if alpha = beta = 0.5, then same as dice
    #   if alpha = beta = 1.0, then same as IoU/Jaccard
    smooth = 1e-5
    
    if y_true.ndim > 3:
        y_true_f = y_true.reshape(y_true.shape[0], -1)
        y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    else:
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
    intersection = torch.sum(y_true_f * y_pred_f)
    tversky = (intersection + smooth) / ( intersection + alpha * (torch.sum(y_pred_f*(1 - y_true_f))) + beta *  (torch.sum((1 - y_pred_f)*y_true_f)) + smooth)
    
    if reduction == 'mean':
        tversky = tversky.mean()
    return tversky


# # Define image transforms/augmentations

# In[80]:


image_transforms = {}

image_transforms['mask'] = T.Compose([
    T.Resize( int(np.floor(IMAGE_SIZE * 1.05)) ),
    T.ToTensor(),
    T.Grayscale(),
    T.CenterCrop(IMAGE_SIZE)
])


# In[81]:


# find mean and std for normalization purposes
transform = [image_transforms['mask'], image_transforms['mask']]
image_dataset = BreastData(image_list_train, transform=transform)
image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=100, shuffle=True)
data_length = len(image_dataset)

features,labels = next(iter(image_dataloader))
print( features)


# In[76]:


subset = next(iter(image_dataloader))[0].permute(1, 0, 2, 3).reshape(1, -1)

mean = subset.mean(-1)
std = subset.std(-1)
print(f'Mean: {mean} --- std: {std}')


# In[ ]:


image_transforms['train'] = T.Compose([
    T.Resize( int(np.floor(IMAGE_SIZE * 1.05)) ),
    T.ToTensor(),
    T.Grayscale(),
    T.CenterCrop(IMAGE_SIZE),
    T.Normalize(mean=mean, std=std),
    T.RandomErasing(p=0.1, scale=(0.01, 0.1), ratio=(0.2, 5)),
])

image_transforms['test'] = T.Compose([
    T.Resize( int(np.floor(IMAGE_SIZE * 1.05)) ),
    T.ToTensor(),
    T.Grayscale(),
    T.CenterCrop(IMAGE_SIZE),
    T.Normalize(mean=mean, std=std),
])


# # Split Image data into Train and Test sets

# In[ ]:


train_transform = [image_transforms['train'], image_transforms['mask']]
test_transform = [image_transforms['test'], image_transforms['mask']]

train_dataset = BreastData(image_list_train, transform=train_transform)
test_dataset = BreastData(image_list_test, transform=test_transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

print(f'Train set size: {len(train_dataset)} ----- Test set size: {len(test_dataset)}')


# # Visual inspection of random data samples

# In[ ]:


images, masks, labels = next(iter(train_dataloader))
labels = np.array(labels)

fig, ax = plt.subplots(4,8, figsize=(20,12))
ax = ax.ravel()
for a in ax:
    a.axis('off')
    
for ind, (image, mask, label) in enumerate( zip(images, masks, labels) ):
    if ind < 16:
        plt.sca(ax[2 * ind])
        plt.imshow(image.permute(1,2,0), cmap='gray')
        plt.title(f'#{ind+1}: {inx_to_class[label]}')
        plt.sca(ax[2 * ind + 1])
        plt.imshow(mask.permute(1,2,0))
        plt.title(f'#{ind+1}')


# # design UNet from scratch

# In[11]:


class unet(nn.Module):
    def __init__(self, CHANNEL_LIST):
        super().__init__()
        self.encoder_block = encoder(CHANNEL_LIST[:-1]) # 128 channels
        self.flat_block = conv_block(CHANNEL_LIST[-2], CHANNEL_LIST[-1]) # 256 channels
        self.decoder_block = decoder(CHANNEL_LIST[::-1][:-1]) # 1 channel
        self.output_block = conv_block(CHANNEL_LIST[1], 1) # 1 channel
        
    def forward(self, x):
        encoder_outputs, x = self.encoder_block(x)
        x = self.flat_block(x)
        x = self.decoder_block(x, encoder_outputs)
        x = f.sigmoid( self.output_block(x) )
        return x    

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)        
        self.dropout = nn.Dropout2d(0.02)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
    
    def forward(self, x):
        x = self.dropout( f.relu( self.batchnorm1( self.conv1(x) ) ) )
        x = self.conv2(x)
        return x
    
    
class encoder(nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        self.blocks = nn.ModuleList([
            conv_block(channel_list[i], channel_list[i+1])
            for i in range(len(channel_list)-1)
        ])
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        encoder_outputs = []
        for block in self.blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.maxpool(x)
        return encoder_outputs, x
    
class decoder(nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose2d(channel_list[i], channel_list[i+1], 2, 2)  
            for i in range(len(channel_list)-1)
        ])
        self.blocks = nn.ModuleList([
            conv_block(channel_list[i], channel_list[i+1])
            for i in range(len(channel_list)-1)
        ])
    
    def forward(self, x, encoder_outputs):
        encoder_outputs_reversed = encoder_outputs[::-1]

        for ind, (upsample, block) in enumerate( zip(self.upsamples, self.blocks) ):            
            x = upsample(x) # doubles the spatial dimension and decrease channels
            x = torch.cat([x, encoder_outputs_reversed[ind]], dim=1) # channels are increased again
            x = block(x) # decrease number of channels
        return x
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)


# # Instantiate the model

# In[12]:


UNet = unet(CHANNEL_LIST).to(device)
initialize_weights(UNet)

criterion = tverskyloss(alpha=0.5, beta=0.5)

optimizer = optim.Adam(UNet.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)


# # Train the model

# In[ ]:


train_losses = []
test_losses = []
train_metrics = []
test_metrics = []
lrs = []

for epoch in range(N_EPOCHS):
    UNet.train()
    
    current_lr = optimizer.state_dict()['param_groups'][0]['lr'] 
    lrs.append( current_lr )
    
    running_loss = 0
    running_metric = 0
    
    for (images, masks, _) in train_dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        output = UNet(images)
        
        # loss 
        loss = criterion(output, masks)
        running_loss += loss.item() * images.shape[0]
        
        # accuracy metric
        img1 = torch.where(output.detach().cpu() >= 0.5, 1, 0)
        img2 = masks.detach().cpu()
        running_metric = running_metric + tversky_index(img1, img2) * images.shape[0]
        
        loss.backward()
        optimizer.step()
        
    lr_scheduler.step( running_loss )

    train_losses.append( running_loss / len(train_dataset) )
    train_metrics.append( running_metric / len(train_dataset) * 100 )
    

    with torch.no_grad():
        UNet.eval()
        
        running_loss = 0
        running_metric = 0
        for (images, masks, _) in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)
        
            output = UNet(images)
            
            # loss
            loss = criterion(output, masks)
            running_loss += loss.item() * images.shape[0]
        
            # accuracy metric
            img1 = torch.where(output.detach().cpu() >= 0.5, 1, 0)
            img2 = masks.detach().cpu()
            running_metric = running_metric + tversky_index(img1, img2) * images.shape[0]
        
        test_losses.append( running_loss / len(test_dataset) )
        test_metrics.append( running_metric / len(test_dataset) * 100 )
        
    print(f'[{epoch+1}/{N_EPOCHS}]: train_loss={train_losses[-1]:0.2f} / test_loss={test_losses[-1]:0.2f} -------------  train_tversky={train_metrics[-1]:0.2f} / test_tversky={test_metrics[-1]:0.2f} -------------  lr={current_lr:0.8f}')


# # Visualizer model performance across epochs

# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(8,12))
ax[0].plot(np.array(train_losses), marker='o', label='Train_loss')
ax[0].plot(np.array(test_losses), marker='o', label='Test_loss')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Losses')

ax[1].plot(np.array(train_metrics), marker='o', label='Train_metric')
ax[1].plot(np.array(test_metrics), marker='o', label='Test_metric')
ax[1].legend()
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Dice index')

ax[2].plot(np.array(lrs), marker='o', label='Train_accuracy')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Learning rate')


# # Save the model

# In[ ]:


torch.save(UNet, '/kaggle/working/trained_model')


# # Visually inspect segmentations

# In[ ]:


fig, ax = plt.subplots(3, 10, figsize=(22, 6))
for this_ax in ax.ravel():
    this_ax.axis('off')
    
for counter, ind in enumerate(np.random.choice(np.arange(len(test_dataset)), 10)):
    image, mask, _ = test_dataset[ind]
    
    ax[0, counter].imshow( image.permute(1,2,0) )
    ax[0, counter].set_title('Breast scan')
    
    ax[1, counter].imshow( mask.permute(1,2,0) )
    ax[1, counter].set_title('Actual area')
    
    segmentation = (UNet(torch.unsqueeze(image, 0).to(device)).detach().cpu() > 0.5)
    ax[2, counter].imshow( torch.squeeze(segmentation, 0).permute(1,2,0) )
    ax[2, counter].set_title('Predicted are')


# # Check for possible biases in model performance per cancer type

# In[ ]:


mask.shape


# In[ ]:


jaccard_index_per_type = {key: [] for key in class_to_inx.keys()}

for (image, mask, label) in test_dataset:
    image = torch.unsqueeze(image, 0)
    mask = torch.unsqueeze(mask, 0)
    
    segmentation = torch.where(UNet(image.to(device)).detach() > 0.5, 1, 0)
    j = tversky_index(segmentation, mask)
    
    key = inx_to_class[label]
    jaccard_index_per_type[key].append( j )

jaccard_index_per_type = {key: np.array(val) for key, val in jaccard_index_per_type.items()}


# In[ ]:


for ind, (key, val) in enumerate(jaccard_index_per_type.items()):
    X = ind + np.random.randn(val.size) * 0.05
    plt.scatter(X, val, alpha=0.5)
    plt.plot([ind-0.2, ind+0.2], [val.mean(), val.mean()], linewidth=5)

plt.xlabel('Cancer type')
plt.ylabel('Jaccard index')

_ = plt.xticks(ticks=[0, 1], labels=jaccard_index_per_type.keys())
percentage = jaccard_index_per_type['benign'].mean() / jaccard_index_per_type['malignant'].mean() * 100

if percentage > 100:
    key_better = 'benign'
else:
    key_better = 'malignant'
percentage = abs( percentage - 100 )
print(f'Model performs better for class "{key_better}" by [{percentage:0.2f}] percent in Dice index.')

