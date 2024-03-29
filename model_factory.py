################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50, ResNet50_Weights
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
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.relu = nn.ReLU()
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
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=outputs)

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

        ''' FC: (64, 128) -> (64, 1024) -> (64, 1024) -> (64, output) '''
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
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

        ''' Functions '''
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

        ''' Models '''
        if self.model_type == 'Custom':
            self.encoder = CustomCNN(self.embedding_size)
        elif self.model_type == 'Resnet':
            self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.encoder.fc = nn.Linear(in_features=512*4, out_features=self.embedding_size)

        self.embedding = nn.Embedding(num_embeddings=self.vocab.idx, embedding_dim=self.embedding_size, padding_idx=0)
        self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab.idx)

    def retrive_caption(self, raw_out):
        batch_size, length, _ = raw_out.size()
        caption_out = torch.zeros((0, length), dtype=torch.long).cuda()

        for i in range(batch_size):
            if self.deterministic:
                pass
                softmax_out = self.softmax(raw_out[i])
                word_out = torch.argmax(softmax_out, 1)
            else:
                softmax_out = self.softmax(raw_out[i] / self.temp)
                word_out = torch.multinomial(softmax_out, 1)

            word_out = word_out.view((1, -1))
            caption_out = torch.cat((caption_out, word_out), 0)

        return caption_out

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
        batch_size, _, _, _ = images.size()

        if captions is not None:
            _, length = captions.size()

            if length < self.max_length:
                zeros = torch.zeros((batch_size, self.max_length - length), dtype=torch.long).cuda()
                captions = torch.cat((captions, zeros), 1)
        input = self.encoder(images).view(-1, 1, self.embedding_size)
        out = torch.zeros((batch_size, 0, self.vocab.idx)).cuda()
        h0 = torch.zeros(2, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(2, batch_size, self.hidden_size).cuda()
        hidden_states = (h0, c0)

        for i in range(self.max_length):
            hidden_out, hidden_states = self.decoder(input, hidden_states)
            raw_out = self.fc(self.flatten(hidden_out)).view((batch_size, 1, self.vocab.idx))
            out = torch.cat((out, raw_out), 1)
            caption_out = self.retrive_caption(raw_out)

            if teacher_forcing:
                input = self.embedding(captions[:, i].view(-1, 1))
            else:
                input = self.embedding(caption_out)

        return out, captions[:, :20]


def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab)