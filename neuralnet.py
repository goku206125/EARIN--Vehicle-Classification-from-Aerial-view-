import torch.nn as nn
import torch.nn.functional as F

# Defines the neural network
class Net(nn.Module):
    def __init__(self, convolution_layer_width=16, kernel1_size=5, kernel2_size=5, linear_layer_width=128, output_layer_width=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, convolution_layer_width, kernel1_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(convolution_layer_width, convolution_layer_width, kernel2_size)
        self.fc1 = nn.Linear(convolution_layer_width * (((32 - kernel1_size + 1) // 2 - kernel2_size + 1) // 2) ** 2, linear_layer_width)
        self.fc2 = nn.Linear(linear_layer_width, linear_layer_width)
        self.fc3 = nn.Linear(linear_layer_width, output_layer_width)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x