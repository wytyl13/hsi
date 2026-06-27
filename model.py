#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/04/27 15:06
@Author  : weiyutao
@File    : model.py
"""

import torch
import torch.nn as nn


class Conv1DSpectralAutoencoder(nn.Module):
    """
    针对高光谱信号一阶导数定制的一维卷积自编码器
    encoder:
        (batch, 1, 203) -> conv(i=1, o=16, kernel_size=7, stride=1, padding=3) -> (batch, 16, 203) -> batch normal -> relu -> maxpool(kernel_size=2) -> (batch, 16, 101)
        -> conv(i=16, o=32, kernel_size=5, stride=1, padding=2) -> batch normal -> relu -> maxpool(kernel_size=2) -> (batch, 32, 50) -> flatten(batch, 32*50)
        -> FCN(32*50, 64) -> relu -> FCN(64, latent_dim)
    decoder:
        FCN(latent_dim, 64) -> relu -> FCN(64, 32*50) -> relu -> unflatten(batch, 32, 50) -> upsample(scale_factor=2) -> (batch, 32, 100) ->
        conv(i=32, o=16, kernel_size=5, stride=1, padding=2) -> (batch, 16, 100) -> batch normal -> relu -> upsample(size=203) -> (batch, 16, 203) ->
        conv(i=16, o=1, kernel_size=7, stride=1, padding=3) -> (batch, 1, 203) 
    """
    
    def __init__(self, input_dim=203, latent_dim=5):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(in_features=32*50, out_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, latent_dim)
        )


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=64, out_features=32*50),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Unflatten(1, (32, 50)),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Upsample(size=input_dim),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=7, stride=1, padding=3)
        )


    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed.view(x.size(0), -1)
    


class WeightedMSALoss(nn.Module):
    """带人工先验的加权均方误差，强制模型关注特定物理波段"""
    def __init__(self, target_indices, weight=50.0):
        super().__init__()
        self.target_indices = target_indices
        self.weight = weight
        self.base_criterion = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss_matrix = self.base_criterion(pred, target)
        weight_mask = torch.ones_like(loss_matrix)
        # 给 1400nm 附近的波段分配极高权重
        weight_mask[:, self.target_indices] = self.weight
        return (loss_matrix * weight_mask).mean()


    


