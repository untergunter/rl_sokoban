import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicQNet(nn.Module):

    def __init__(self, input_size, hidden_size,outputs:tuple):
        self.actions = outputs
        super(BasicQNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_next_step(self,x,device=None):
        all_outputs = torch.tensor([np.concatenate([x,np.array([action])]) for action in self.actions]).float()
        if device:
            all_outputs = all_outputs.to(device)
        prob_dist = self.forward(all_outputs)
        pred = self.actions[int(torch.argmax(prob_dist))]
        return pred


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNet(nn.Module):

    def __init__(self, input_size, hidden_size,outputs:tuple):
        self.actions = outputs
        super(DeepQNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calculate_next_step(self,x,device=None):
        all_outputs = torch.tensor([np.concatenate([x,np.array([action])]) for action in self.actions]).float()
        if device:
            all_outputs = all_outputs.to(device)
        prob_dist = self.forward(all_outputs)
        pred = self.actions[int(torch.argmax(prob_dist))]
        return pred

if __name__=='__main__':
    state = np.arange(3)
    n = BasicQNet(4, 2, (1, 2))
    print(n.calculate_next_step(state))

if __name__=='__main__':
    state = np.arange(3)
    n = BasicQNet(4, 2, (1, 2))
    print(n.calculate_next_step(state))