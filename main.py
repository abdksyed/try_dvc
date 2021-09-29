import os
from pathlib import Path
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import math
from tqdm import tqdm

import torch.backends.cudnn as cudnn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16_bn
# Downloading: "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth" to C:\Users\<USERNAME>/.cache\torch\hub\checkpoints\vgg16_bn-6c64b313.pth #Windows
# Downloading: "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth" to /home/<USERNAME>/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth #Linux(WSL2)

import numpy as np
import random

from bottleneck_features import save_feature_map
from top_model import ModelHead
from trainer import train, test

logging.basicConfig(filename='cat_dog.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.info('Starting')

csvfile = open('metrics.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=',')
csvwriter.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy',
                         'Cat Accuracy', 'Dog Accuracy'])

cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dimensions of our images.
img_width, img_height = 224, 224


pathname = os.path.dirname(sys.argv[0]) #Folder of py file being run
path = os.path.abspath(pathname) #Absolute Path of the the file
path = Path(path)

top_model_weights_path = 'model.h5'
train_data_dir = Path('data').joinpath('train')
validation_data_dir = Path('data').joinpath('validation')
cats_train_path = path/train_data_dir/'cats'

transformations = {
        'train': transforms.Compose([
            transforms.Resize([img_width,img_height]), # Resizing the image
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]),
        'test': transforms.Compose([
            transforms.Resize([img_width,img_height]),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])
    }

train_data = ImageFolder(str(train_data_dir), transform=transformations['train'])
test_data = ImageFolder(str(validation_data_dir), transform=transformations['test'])


# Creating the dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


model = vgg16_bn(pretrained=True)
feature_extractor = nn.Sequential(*list(model.children())[:-2])

logger.info('Extracting Training Data Features, from VGG16_BN')
save_feature_map(train_loader, feature_extractor, path, 'train', logger, device,force=True)
logger.info('Extracting Test Data Features, from VGG16_BN')
save_feature_map(test_loader, feature_extractor, path, 'test', logger, device, force=True)
    
train_data = np.load(open('train_bottleneck_features.npy', 'rb'))
train_labels = np.load(open('train_labels.npy', 'rb'))

test_data = np.load(open('test_bottleneck_features.npy', 'rb'))
test_labels = np.load(open('test_labels.npy', 'rb'))

modelhead = ModelHead(math.prod(train_data[0].shape), 8, 1)
optim = optim.SGD(modelhead.parameters(), lr=0.0001, momentum=0.9)

for epoch in tqdm(range(10)):
    logger.info('Epoch: {}'.format(epoch))
    train_loss, train_acc = train(modelhead, train_data, train_labels, 16, optim, epoch, logger, device)
    test_loss, test_acc, (cat_acc, dog_acc) = test(modelhead, test_data, test_labels, 16, epoch, logger, device)
    csvwriter.writerow([epoch, round(train_loss,4), train_acc, round(test_loss,4), test_acc, cat_acc, dog_acc])


csvfile.close()
top_model_weights_path = Path('model.h5')
logger.info('Saving the Head Model Weights as "model.h5"...')
torch.save(modelhead.state_dict(), top_model_weights_path)
logger.info("Model Saved!")