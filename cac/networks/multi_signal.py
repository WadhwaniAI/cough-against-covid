import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from cac.networks.backbones.resnet import resnet18
from cac.networks.backbones.tab_network import TabNet

# MLP(Context) + Resnet+FCLayers(Cough) [Merge at 128]
class NaiveCoughContextNetwork(nn.Module):
    def __init__(self, input_dim_text = 11, dropout = 0.4, merge_type = 'sum'):
        super(NaiveCoughContextNetwork, self).__init__()
        self.merge_type = merge_type
        assert merge_type in ['sum', 'max', 'min'], "Mention Merge Type out of the 3 (Max, Min, Sum)"

        # cough model
        self.resnet_cough = resnet18(in_channels=1, pretrained=True)
        self.adaptiveavgpool_cough = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_cough = nn.Flatten()
        
        self.fc1_cough = nn.Linear(512, 256)
        self.dropout1_cough = nn.Dropout(p=dropout)
        self.relu1_cough = nn.ReLU()
        self.fc2_cough = nn.Linear(256, 128)
        self.dropout2_cough = nn.Dropout(p=dropout)        
        self.relu2_cough = nn.ReLU()        
        self.fc3_cough = nn.Linear(128, 2)
        self.dropout3_cough = nn.Dropout(p=dropout)    
         
        # context MLP
        self.linear1_context = nn.Linear(11, 128)
        self.relu1_context = nn.ReLU()
        self.linear2_context = nn.Linear(128, 128)
        self.relu2_context = nn.ReLU()


    def forward(self, signals, context_signal):
        cough_signal = signals[0]

        # Forward Pass over Cough Signal
        cough_signal = self.adaptiveavgpool_cough(self.resnet_cough(cough_signal))
        cough_signal = self.dropout1_cough(self.flatten_cough(cough_signal))

        cough_signal = self.dropout2_cough(self.relu1_cough(self.fc1_cough(cough_signal)))
        cough_signal = self.dropout3_cough(self.relu2_cough(self.fc2_cough(cough_signal)))

        # Forward Pass over Context Signal
        context_signal = self.relu1_context(self.linear1_context(context_signal))
        context_signal = self.relu2_context(self.linear2_context(context_signal))

        # Naive merge at 128 Level
        if self.merge_type == 'sum':
            cough_signal = torch.sum(torch.stack([cough_signal, context_signal], dim = -1), dim = 2)
        elif self.merge_type == 'max':
            cough_signal, _ = torch.max(torch.stack([cough_signal, context_signal], dim = -1), dim = 2)
        elif self.merge_type == 'min':
            cough_signal = torch.min(torch.stack([cough_signal, context_signal], dim = -1), dim = 2)
        else: 
            raise AssertionError

        cough_signal = self.fc3_cough(cough_signal)        
        return cough_signal

# Resnet+FCLayers(Cough) + Resnet+FCLayers(Voice) [Merge at 128]
class NaiveCoughVoiceNetwork(nn.Module):
    def __init__(self, input_dim_text = 11, dropout = 0.4, merge_type = 'sum'):
        super(NaiveCoughVoiceNetwork, self).__init__()
        self.merge_type = merge_type
        assert merge_type in ['sum', 'max', 'min'], "Mention Merge Type out of the 3 (Max, Min, Sum)"

        # cough model
        self.resnet_cough = resnet18(in_channels=1, pretrained=True)
        self.adaptiveavgpool_cough = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_cough = nn.Flatten()
        
        self.fc1_cough = nn.Linear(512, 256)
        self.dropout1_cough = nn.Dropout(p=dropout)
        self.relu1_cough = nn.ReLU()
        self.fc2_cough = nn.Linear(256, 128)
        self.dropout2_cough = nn.Dropout(p=dropout)        
        self.relu2_cough = nn.ReLU()        
        self.fc3_cough = nn.Linear(128, 2)
        self.dropout3_cough = nn.Dropout(p=dropout)    
         
        # voice model
        self.resnet_voice = resnet18(in_channels=1, pretrained=True)
        self.adaptiveavgpool_voice = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_voice = nn.Flatten()
        
        self.fc1_voice = nn.Linear(512, 256)
        self.dropout1_voice = nn.Dropout(p=dropout)
        self.relu1_voice = nn.ReLU()
        self.fc2_voice = nn.Linear(256, 128)
        self.dropout2_voice = nn.Dropout(p=dropout)        
        self.relu2_voice = nn.ReLU()        
        self.dropout3_voice = nn.Dropout(p=dropout) 


    def forward(self, signals, context_signal):
        cough_signal = signals[0]
        voice_signal = signals[1]
        
        # Forward Pass over Cough Signal
        cough_signal = self.adaptiveavgpool_cough(self.resnet_cough(cough_signal))
        cough_signal = self.dropout1_cough(self.flatten_cough(cough_signal))

        cough_signal = self.dropout2_cough(self.relu1_cough(self.fc1_cough(cough_signal)))
        cough_signal = self.dropout3_cough(self.relu2_cough(self.fc2_cough(cough_signal)))

        # Forward Pass over Voice Signal
        voice_signal = self.adaptiveavgpool_voice(self.resnet_cough(voice_signal))
        voice_signal = self.dropout1_voice(self.flatten_voice(voice_signal))

        voice_signal = self.dropout2_voice(self.relu1_voice(self.fc1_voice(voice_signal)))
        voice_signal = self.dropout3_voice(self.relu2_voice(self.fc2_voice(voice_signal)))

        # Naive merge at 128 Level
        if self.merge_type == 'sum':
            cough_signal = torch.sum(torch.stack([cough_signal, voice_signal], dim = -1), dim = 2)
        elif self.merge_type == 'max':
            cough_signal, _ = torch.max(torch.stack([cough_signal, voice_signal], dim = -1), dim = 2)
        elif self.merge_type == 'min':
            cough_signal = torch.min(torch.stack([cough_signal, voice_signal], dim = -1), dim = 2)
        else: 
            raise AssertionError

        cough_signal = self.fc3_cough(cough_signal)        
        return cough_signal

# Resnet+FCLayers(Cough) + Resnet+FCLayers(Voice) + MLP(Context) [Merge at 128]
class NaiveCoughVoiceContextNetwork(nn.Module):
    def __init__(self, input_dim_text = 11, dropout = 0.4, merge_type = 'sum'):
        super(NaiveCoughVoiceContextNetwork, self).__init__()
        self.merge_type = merge_type
        assert merge_type in ['sum', 'max', 'min'], "Mention Merge Type out of the 3 (Max, Min, Sum)"

        # cough model
        self.resnet_cough = resnet18(in_channels=1, pretrained=True)
        self.adaptiveavgpool_cough = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_cough = nn.Flatten()
        
        self.fc1_cough = nn.Linear(512, 256)
        self.dropout1_cough = nn.Dropout(p=dropout)
        self.relu1_cough = nn.ReLU()
        self.fc2_cough = nn.Linear(256, 128)
        self.dropout2_cough = nn.Dropout(p=dropout)        
        self.relu2_cough = nn.ReLU()        
        self.fc3_cough = nn.Linear(128, 2)
        self.dropout3_cough = nn.Dropout(p=dropout)    
         
        # voice model
        self.resnet_voice = resnet18(in_channels=1, pretrained=True)
        self.adaptiveavgpool_voice = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_voice = nn.Flatten()
        
        self.fc1_voice = nn.Linear(512, 256)
        self.dropout1_voice = nn.Dropout(p=dropout)
        self.relu1_voice = nn.ReLU()
        self.fc2_voice = nn.Linear(256, 128)
        self.dropout2_voice = nn.Dropout(p=dropout)        
        self.relu2_voice = nn.ReLU()        
        self.dropout3_voice = nn.Dropout(p=dropout) 

        # context MLP
        self.linear1_context = nn.Linear(11, 128)
        self.relu1_context = nn.ReLU()
        self.linear2_context = nn.Linear(128, 128)
        self.relu2_context = nn.ReLU()

    def forward(self, signals, context_signal):
        cough_signal = signals[0]
        voice_signal = signals[1]
        
        # Forward Pass over Cough Signal
        cough_signal = self.adaptiveavgpool_cough(self.resnet_cough(cough_signal))
        cough_signal = self.dropout1_cough(self.flatten_cough(cough_signal))

        cough_signal = self.dropout2_cough(self.relu1_cough(self.fc1_cough(cough_signal)))
        cough_signal = self.dropout3_cough(self.relu2_cough(self.fc2_cough(cough_signal)))

        # Forward Pass over Voice Signal
        voice_signal = self.adaptiveavgpool_voice(self.resnet_cough(voice_signal))
        voice_signal = self.dropout1_voice(self.flatten_voice(voice_signal))

        voice_signal = self.dropout2_voice(self.relu1_voice(self.fc1_voice(voice_signal)))
        voice_signal = self.dropout3_voice(self.relu2_voice(self.fc2_voice(voice_signal)))

        # Forward Pass over Context Signal
        context_signal = self.relu1_context(self.linear1_context(context_signal))
        context_signal = self.relu2_context(self.linear2_context(context_signal))

        # Naive merge at 128 Level
        if self.merge_type == 'sum':
            cough_signal = torch.sum(torch.stack([cough_signal, voice_signal, context_signal], dim = -1), dim = 2)
        elif self.merge_type == 'max':
            cough_signal, _ = torch.max(torch.stack([cough_signal, voice_signal, context_signal], dim = -1), dim = 2)
        elif self.merge_type == 'min':
            cough_signal = torch.min(torch.stack([cough_signal, voice_signal, context_signal], dim = -1), dim = 2)
        else: 
            raise AssertionError

        cough_signal = self.fc3_cough(cough_signal)        
        return cough_signal

