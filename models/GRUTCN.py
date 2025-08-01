import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
class EngagementPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EngagementPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, num_segments, feature_dim]
        gru_out, _ = self.gru(x)  # [batch_size, num_segments, hidden_dim]
        engagement_values = self.fc(gru_out)  # [batch_size, num_segments, output_dim]
        engagement_values = engagement_values.squeeze(-1)  # [batch_size, num_segments]
        final_engagement = self.mean_pooling(engagement_values.unsqueeze(-1)).squeeze(-1)  # [batch_size]
        return final_engagement


input_dim = 117
hidden_dim = 512
output_dim = 1
model = EngagementPredictor(input_dim, hidden_dim, output_dim)
print(model)