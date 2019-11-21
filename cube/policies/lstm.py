import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, bf=1.0, bn=0.0, device="cpu"):
        super(LSTM, self).__init__()


        # policy parameters
        self.dim_x = input_size
        self.dim_y = output_size
        self.dim_h = hidden_size

        self.f_forget = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.i_input = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.j_input = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.o_output = nn.Linear(input_size+hidden_size, hidden_size, bias=False)

        self.hid2y = nn.Linear(hidden_size, output_size)
    
        self.device = device
        self.bf = torch.tensor(bf).to(device)
        self.bi = torch.tensor(bn).to(device)
        self.bj = torch.tensor(bn).to(device)
        self.bo = torch.tensor(bn).to(device)
        self.cell = self.init_cell().to(device)
        self.h = self.init_hidden().to(device)

    def forward(self, x, h=None, temp=0.1, dropout_rate=0.0, training=False):

        if h is None:
            h = self.h

        training_mode = training


        combined_input = torch.cat([x, h], 1)

        forget = torch.sigmoid(self.f_forget(combined_input) + self.bf)
        self.cell = torch.mul(forget, self.cell)

        remember = torch.sigmoid(self.i_input(combined_input)+self.bi)
        input_states = torch.tanh(self.j_input(combined_input)+self.bj)

        self.cell = self.cell + torch.mul(remember, input_states)

        output = torch.sigmoid(self.o_output(combined_input) + self.bo)

        h = torch.mul(torch.tanh(self.cell), output)
        y = self.hid2y(h)

        y = F.softmax(F.relu(y/temp),\
                dim=1)

        self.h = h

        return y, h

    def get_actions(self, x, h=None, temp=0.1, dropout_rate=0.0, training=False):

        probs, h = self.forward(x, h, temp=temp, dropout_rate=dropout_rate, training=False)
        
        action = torch.multinomial(probs,1)

        return action, probs, h 

    def init_hidden(self, batch_size=1):

        return torch.zeros((batch_size, self.dim_h), device=self.device)

    def init_cell(self, batch_size=1):

        return torch.zeros((batch_size, self.dim_h), device=self.device)
    
if __name__ == "__main__":
    hid_dim = 512 
    obs_dim = 36
    act_dim = 12

    policy = LSTM(obs_dim, act_dim, hid_dim)


    x = np.random.randn(1,1000,obs_dim) + 1e-1 * np.arange(1000)[np.newaxis,:,np.newaxis] 
    y = np.random.randn(1,1000,act_dim) + 1e-1 * np.arange(1000)[np.newaxis,:,np.newaxis]
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    policy.zero_grad()
    loss = 0.0
    for jj in range(10):
        for ii in range(x.shape[1]):
            action, y_pred, h = policy.get_actions(x[:,ii])
            loss = torch.mean((y_pred - y[:,ii,:])**2)

            if ii % x.shape[1] == 0:
                print("loss at epoch {}: {}".format(jj,loss))
                loss.backward()
                optimizer.step()
                policy.h = policy.init_hidden()
                policy.cell = policy.init_cell()
                policy.zero_grad()
                loss = 0.0
