import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

class CNNForecaster(nn.Module):
    
    def __init__(self, input_shape, output_shape, num_layers, dropout, embedding_size=32):
        super(CNNForecaster, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        
        self.embedding = nn.Linear(input_shape[0] * input_shape[1], input_shape[1] * self.embedding_size, bias=False)
                
        self.conv_layers = nn.ModuleList()

        # self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
        for i in range(num_layers):
            self.conv_layers.append(nn.Conv1d(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, dilation=1))
            # self.conv_layers.append(nn.ReLU())
            # self.conv_layers.append(self.max_pool)
        
        self.conv_seq = nn.Sequential(*self.conv_layers)
        
        self.fc = nn.Linear(self.embedding_size * input_shape[1], output_shape[0] * output_shape[1])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        x = x.flatten(start_dim=1)
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.view(embedded_x.size(0), self.embedding_size, self.input_shape[1])
        
        output = self.conv_seq(embedded_x)
        
        output = output.flatten(start_dim=1)
        # output = self.dropout(output)
        output = self.fc(output)
        
        return output

    def reshape_output(self, output):
        """
        Reshapes the output to the original shape
        """
        output = output.view(output.size(0), self.output_shape[0], self.output_shape[1])
        return output
    
if __name__ == "__main__":
    """
    Initialize the model
    
    create a model with the following parameters:
    input shape: (2, 8)
    output shape: (2, 8)
    num_layers: 4
    """
    times = []
    model = CNNForecaster((4, 20), (4, 1), 4, 0.5, 32)
    
    for i in range(10000):
        test_tensor = torch.randn(32, 4, 20)
        
        start = datetime.now()
        output = model(test_tensor)
        output = model.reshape_output(output)
        times.append(datetime.now() - start)
    
    print(np.mean(times))
        
        
            