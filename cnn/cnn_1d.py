import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class cnn_1d(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=10, stride=1)
        self.fc1 = nn.Linear(3536, 3536*4) # (n , n*4) where n=conv2.shape[0]*conv2.shape[1]
        self.fc2 = nn.Linear(3536*4, 7)
        self.activation = activation()
    
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
    

if __name__ == '__main__':
    print()
    input_data = torch.randn((1,234), device=device)
    model = cnn_1d()
    model.to(device)
    model(input_data)