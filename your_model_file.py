import torch
import torch.nn as nn
from torchvision import models

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DeepFakeDetector, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])  # remove last two layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):  # x: (B, T, C, H, W)
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.cnn(x)  # (B*T, 2048, 7, 7)
        x = self.avgpool(features)  # (B*T, 2048, 1, 1)
        x = x.view(batch_size, seq_length, 2048)
        lstm_out, _ = self.lstm(x)
        out = self.fc(torch.mean(lstm_out, dim=1))
        return out
