import torch
from torch import nn

IMG_SIZE = 224

class CatDogModel(nn.Module):
    def __init__(self,input_shape, hidden_units, output_shape,dropout_prob = 0.2):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels = hidden_units*2,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.Conv2d(in_channels = hidden_units*2,
                      out_channels = hidden_units*2,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2,
                      out_channels = hidden_units*4,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.Conv2d(in_channels = hidden_units*4,
                      out_channels = hidden_units*4,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.MaxPool2d(kernel_size = 2)
        )
        flatten_size = self.initialize_classifier(input_shape)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=flatten_size,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=4096,
                      out_features=1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=1024,
                      out_features=output_shape),
        )

    def initialize_classifier(self, input_shape):
        dummy_input = torch.zeros(1, input_shape, IMG_SIZE, IMG_SIZE)
        dummy_output = self.conv_block_1(dummy_input)
        dummy_output = self.conv_block_2(dummy_output)
        dummy_output = self.conv_block_3(dummy_output)
        dummy_output = self.conv_block_4(dummy_output)
        dummy_output = self.conv_block_5(dummy_output)
        flatten_size = dummy_output.numel()
        print(flatten_size)
        return flatten_size

    def forward(self,x):
        x = self.conv_block_1(x)
        x= self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.classifier(x)
        return x
    
"""
model = CatDogModel(input_shape=3,
                        hidden_units=32,
                        output_shape=1,
                    dropout_prob = 0.5
                    ).to(device)
"""