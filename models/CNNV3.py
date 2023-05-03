import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNForecaster(nn.Module):
    
    def __init__(self, input_shape, output_shape, dropout, conv_dim=32):
        super(CNNForecaster, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = input_shape[1] // 2
        self.dropout = dropout
        self.conv_dim = conv_dim
                        
        self.conv_layers = nn.ModuleList()

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.conv_dim, kernel_size=3, stride=1, padding=1)
        
        for i in range(1, self.num_layers):
            self.conv_layers.append(nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(self.max_pool)
        
        self.conv_seq = nn.Sequential(*self.conv_layers)
        
        self.fc = nn.Linear(self.conv_dim * input_shape[1], self.input_shape[1])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x1, x2, x3, x4 = x[:, 0, :], x[:, 1, :], x[:, 2, :], x[:, 3, :]
        x1, x2, x3, x4 = x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)
        
        x1 = self.max_pool(F.relu(self.conv1(x1)))
        x1 = self.conv_seq(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        
        x2 = self.max_pool(F.relu(self.conv1(x2)))
        x2 = self.conv_seq(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)
        
        x3 = self.max_pool(F.relu(self.conv1(x3)))
        x3 = self.conv_seq(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc(x3)
        
        x4 = self.max_pool(F.relu(self.conv1(x4)))
        x4 = self.conv_seq(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.fc(x4)
        
        x1, x2, x3, x4 = x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)
        
        output = torch.cat((x1, x2, x3, x4), dim=1)
        
        return output
    
if __name__ == "__main__":
    """
    Initialize the model
    
    create a model with the following parameters:
    input shape: (2, 8)
    output shape: (2, 8)
    num_layers: 4
    """
    model = CNNForecaster((4, 10), (4, 10), 0.5, 32)
    test_tensor = torch.randn(32, 4, 10)
    
    output = model(test_tensor)
    print(output.shape)
        
        
            