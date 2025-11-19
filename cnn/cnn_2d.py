import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class cnn_2d(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(420, 420*4)
        self.fc2 = nn.Linear(420*4, 7)
        self.activation = activation()

        pass
    
    def forward(self, x):
        # convolutional layers
        print(f"Input: {x.shape}")
        x = self.conv1(x)
        print(f'conv1 => {x.shape}')
        x = self.activation(x)
        x = self.conv2(x)
        print(f'conv2 => {x.shape}')

        # Flatten the conv2 output for the subsequent linear layers
        x = x.view(-1, x.shape[0] * x.shape[1])

        # fully-connected layers
        print(f'{x.shape} => fc1')
        x = self.activation(x)
        x = self.fc1(x)
        print(f'fc1 => {x.shape}')
        x = self.activation(x)
        x = self.fc2(x)
        print(f'fc2 => {x.shape}')
        return x
    
class cnn_2d_maxpool(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=4, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3, stride=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # depending on the input size it may not always make sense to have many max pool layers

        self.flatten = nn.Flatten(0)

        self.fc1 = nn.Linear(196, 196*4)
        self.fc2 = nn.Linear(196*4, 7)
        
        self.activation = activation()

        pass
    
    def forward(self, x):
        # convolutional layers
        print(f"Input: {x.shape}")
        x = self.conv1(x)
        print(f'conv1 => {x.shape}')
        x = self.activation(x)
        x = self.max_pool1(x)
        print(f'max_pool1 => {x.shape}')
        x = self.conv2(x)
        print(f'conv2 => {x.shape}')

        # x = self.max_pool2(x)
        # print(f'max_pool2 => {x.shape}')

        # Flatten the conv2 output for the subsequent linear layers
        x = self.flatten(x).unsqueeze(0)
        # x = x.view(-1, x.shape[0] * x.shape[1])

        # fully-connected layers
        print(f'{x.shape} => fc1')
        x = self.activation(x)
        x = self.fc1(x)
        print(f'fc1 => {x.shape}')
        x = self.activation(x)
        x = self.fc2(x)
        print(f'fc2 => {x.shape}')
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        # Padding is set to maintain height and width.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.activation = activation()

    def forward(self, x):
        residual = x
        out = self.activation(self.conv(x))
        # Add the residual connection
        out += residual
        return self.activation(out)
    
class cnn_2d_maxres(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=4, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=3, stride=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # depending on the input size it may not always make sense to have many max pool layers

        self.flatten = nn.Flatten(0)

        self.fc1 = nn.Linear(196, 196*4)
        self.fc2 = nn.Linear(196*4, 7)
        
        self.activation = activation()

        pass
    
    def forward(self, x):
        # convolutional layers
        print(f"Input: {x.shape}")
        x = self.conv1(x)
        print(f'conv1 => {x.shape}')
        x = self.activation(x)
        x = self.max_pool1(x)
        print(f'max_pool1 => {x.shape}')
        x = self.conv2(x)
        print(f'conv2 => {x.shape}')

        # x = self.max_pool2(x)
        # print(f'max_pool2 => {x.shape}')

        # Flatten the conv2 output for the subsequent linear layers
        x = self.flatten(x).unsqueeze(0)
        # x = x.view(-1, x.shape[0] * x.shape[1])

        # fully-connected layers
        print(f'{x.shape} => fc1')
        x = self.activation(x)
        x = self.fc1(x)
        print(f'fc1 => {x.shape}')
        x = self.activation(x)
        x = self.fc2(x)
        print(f'fc2 => {x.shape}')
        return x

if __name__ == '__main__':
    print()
    input_data = torch.randn((1,21,10), device=device) # shape = (C, H, W) of an image, where C must be specified
    model = cnn_2d()
    model.to(device)
    model(input_data)

    print()
    model = cnn_2d_maxpool()
    model.to(device)
    model(input_data)