from classifier_models.vgg import *
from classifier_models import vgg_tiny_imagenet
import torchvision
import torch
from classifier_models.resnet_tinyimagenet import resnet18
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import copy

device = None
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, heatmap = None):
        x = self.encoder(x)
        x = self.decoder(x)
        if heatmap != None:
          heatmap = heatmap.to(device = device)
          temp_zero = torch.zeros_like(x).to(device = device)
          x = torch.where(heatmap > 0, x, temp_zero)
        return x

def clip_image(x):
    return torch.clamp(x, 0, 1.0)