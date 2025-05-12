import torch.nn as nn

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.sub_model = SubModel()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.sub_model(x)
        return self.fc2(x)
