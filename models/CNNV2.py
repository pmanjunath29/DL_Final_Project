import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNForecaster(nn.Module):
    
    def __init__(self, input_shape, output_shape):
        super(CNNForecaster, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        
        self.linear = nn.Linear(16 * , 100)
        self.final_fc = nn.Linear(100, self.output_shape[0] * self.output_shape[1])
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        output = self.final_fc(x)
        
        output = output.view(output.size(0), self.output_shape[0], self.output_shape[1])
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
    model = CNNForecaster((4, 10), (4, 1))
    test_tensor = torch.randn(32, 4, 10)
    
    output = model(test_tensor)
    output = model.reshape_output(output)
    print(output.shape)
        
        
            