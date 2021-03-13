#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, GRU, Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation

__author__ = 'Helin Wang -- Peking University'
__docformat__ = 'reStructuredText'
__all__ = ['Encoder','CNNEncoder','CNNEncoder2']

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

class CNNEncoder(nn.Module):
    
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        #self.fc = nn.Linear(512, 512, bias=True)

    def forward(self, input):
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        #x = self.fc(x)
        x = torch.mean(x, dim=2)
        
        return x

class CNNEncoder2(nn.Module):
    
    def __init__(self):
        super(CNNEncoder2, self).__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=4, 
            freq_drop_width=8, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        #self.fc = nn.Linear(512, 512, bias=True)

    def forward(self, input):
        x = input[:, None, :, :]
        if self.training:
            x = self.spec_augmenter(x)
        x = F.avg_pool2d(x, kernel_size=(10,1), stride=(5,1))
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        #x = x.transpose(1,2)
        #x = x.contiguous().view(x.shape[0],x.shape[1],-1)  # (batch_size, feature_maps, time_stpes)
        #x = self.fc(x)
        #x = x.transpose(1,2)
        x = torch.mean(x, dim=3)
        
        return x

class Encoder(Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_p: float) \
            -> None:
        """Encoder module.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param hidden_dim: Hidden dimensionality.
        :type hidden_dim: int
        :param output_dim: Output dimensionality.
        :type output_dim: int
        :param dropout_p: Dropout.
        :type dropout_p: float
        """
        super(Encoder, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        self.dropout: Module = Dropout(p=dropout_p)

        rnn_common_args = {
            'num_layers': 1,
            'bias': True,
            'batch_first': True,
            'bidirectional': True}

        self.gru_1: Module = GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            **rnn_common_args)

        self.gru_2: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.hidden_dim,
            **rnn_common_args)

        self.gru_3: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.output_dim,
            **rnn_common_args)

    def _l_pass(self,
                layer: Module,
                layer_input: Tensor) \
            -> Tensor:
        """Does the forward passing for a GRU layer.

        :param layer: GRU layer for forward passing.
        :type layer: torch.nn.Module
        :param layer_input: Input to the GRU layer.
        :type layer_input: torch.Tensor
        :return: Output of the GRU layer.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = layer_input.size()
        h = layer(layer_input)[0].view(b_size, t_steps, 2, -1)
        return self.dropout(cat([h[:, :, 0, :], h[:, :, 1, :]], dim=-1))

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """
        h = self._l_pass(self.gru_1, x)

        for a_layer in [self.gru_2, self.gru_3]:
            h_ = self._l_pass(a_layer, h)
            h = h + h_ if h.size()[-1] == h_.size()[-1] else h_

        return h

# EOF
