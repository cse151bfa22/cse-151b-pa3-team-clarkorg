################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        ''' Functions '''
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

        ''' Layer 1 '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        ''' Layer 2 '''
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=128)

        ''' Layer 3 '''
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)

        ''' Layer 4 '''
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)

        ''' Layer 5 '''
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128)

        ''' Fully Connected Layers '''
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, outputs)

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        ''' Input: (64, 3, 256, 256) '''

        ''' Layer 1: (64, 64, 30, 30) '''
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        ''' Layer 2: (64, 128, 14, 14) '''
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.maxpool(out)

        ''' Layer 3: (64, 256, 14, 14) '''
        out = self.relu(self.bn3(self.conv3(out)))

        ''' Layer 4: (64, 256, 14, 14) '''
        out = self.relu(self.bn4(self.conv4(out)))

        ''' Layer 5: (64, 128, 6, 6) '''
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.maxpool(out)

        ''' Flatten: (64, 128) '''
        out = self.avgpool(out)
        out = self.flatten(out)

        ''' FC: (64, 128) -> (64, 1024) -> (64, 1024) -> (64, output)'''
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out



class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']

        # TODO
        raise NotImplementedError()


    def forward(self, images, captions, teacher_forcing=False):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        # TODO
        raise NotImplementedError()


def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    # return CNN_LSTM(config_data, vocab)
    return CustomCNN(100)