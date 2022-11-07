import torch
import torch.nn as nn
import torch.nn.functional as F



class LeNet5Like(nn.Module):

    def __init__(self, dropout_rate=0.05):
        super(LeNet5Like, self).__init__()
        
        self.encoder = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84), 
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        res = F.softmax(x, dim=1)
        return res



class LeNet5Like1(nn.Module):

    def __init__(self):
        super(LeNet5Like1, self).__init__()
        
        self.encoder = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=80, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=80, out_channels=120, kernel_size=2, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84), 
            nn.Dropout(0.05),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        res = F.softmax(x, dim=1)
        return res



class LeNet5Like2(nn.Module):

    def __init__(self):
        super(LeNet5Like2, self).__init__()
        
        self.encoder = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=60, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=60, out_features=84), 
            nn.Dropout(0.05),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        res = F.softmax(x, dim=1)
        return res




class LeNet5Like3(nn.Module):

    def __init__(self):
        super(LeNet5Like3, self).__init__()
        
        self.encoder = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=240, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=240, out_features=84), 
            nn.Dropout(0.05),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        res = F.softmax(x, dim=1)
        return res

class LeNet5Like4(nn.Module):

    def __init__(self):
        super(LeNet5Like4, self).__init__()
        
        self.encoder = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84), 
            nn.Dropout(0.05),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        res = F.softmax(x, dim=1)
        return res

