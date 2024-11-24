import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_bandit_alg import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype_cuda = torch.bfloat16

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, dtype=dtype_cuda)
        self.affine2 = nn.Linear(hidden_size, out_size, dtype=dtype_cuda)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)

class ReplayBuffer:

    def __init__(self, d, capacity):
        self.buffer = {'context':np.zeros((capacity, d)), 'reward': np.zeros((capacity,1))}
        self.capacity = capacity
        self.size = 0
        self.pointer = 0


    def add(self, context, reward):
        self.buffer['context'][self.pointer] = context
        self.buffer['reward'][self.pointer] = reward
        self.size = min(self.size+1, self.capacity)
        self.pointer = (self.pointer+1)%self.capacity

    def sample(self, n):
        idx = np.random.randint(0,self.size,size=n)
        return self.buffer['context'][idx], self.buffer['reward'][idx]

class NeuralUCB(RandomAlg):

    def __init__(self, d, K, beta=1, lamb=1, hidden_size=128, lr=1e-4, reg=0.000625):
        self.K = K
        self.T = 0
        self.reg = reg
        self.beta = beta
        self.device = device  # Ensure device is set before using it

        self.net = Model(d, hidden_size, 1)
        self.hidden_size = hidden_size
        self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.numel = sum(w.numel() for w in self.net.parameters() if w.requires_grad)

        # Initialize sigma_inv as a PyTorch tensor on the GPU
        self.sigma_inv = lamb * torch.eye(self.numel, dtype=dtype_cuda, device=self.device)

        # Ensure theta0 is on the correct device
        self.theta0 = torch.cat(
            [w.flatten() for w in self.net.parameters() if w.requires_grad]
        ).to(self.device)

        self.replay_buffer = ReplayBuffer(d, 10000)

    def take_action(self, context):
        # Move context tensor directly to the device
        context = torch.tensor(context, dtype=dtype_cuda, device=self.device)
        context = context.unsqueeze(0).expand(self.K, -1)

        # Initialize g as a tensor on the device
        g = torch.zeros((self.K, self.numel), dtype=dtype_cuda, device=self.device)

        # Compute gradients for each action
        for k in range(self.K):
            g[k] = self.grad(context[k]).detach()

        with torch.no_grad():
            # Compute the model output
            pred = self.net(context).squeeze()  # Shape: [K]

            # Compute the UCB term
            gs = torch.matmul(g, self.sigma_inv)  # Shape: [K, numel]
            uncertainty = (gs * g).sum(dim=1)  # Shape: [K]
            ucb_term = self.beta * torch.sqrt(uncertainty)

            # Compute the final scores
            p = pred + ucb_term  # Shape: [K]

        # Select the action with the highest score
        action = torch.argmax(p).item()
        return action

    def grad(self, x):
        # Ensure x is a tensor on the correct device
        x = x.to(self.device).unsqueeze(0)  # Shape: [1, d]
        y = self.net(x)
        self.optimizer.zero_grad()
        y.backward()

        # Concatenate gradients into a single vector
        grad_list = [
            w.grad.flatten() / torch.sqrt(torch.tensor(self.hidden_size, dtype=dtype_cuda))
            for w in self.net.parameters() if w.requires_grad
        ]
        grad_vector = torch.cat(grad_list).to(self.device)
        return grad_vector

    def update(self, context, action, reward):
        # Move context tensor directly to the device
        context = torch.tensor(context, dtype=torch.float32, dtype=dtype_cuda)
        context = context.unsqueeze(0).expand(self.K, -1)

        # Compute the gradient for the selected action
        v = self.grad(context[action]).detach()

        # Update sigma_inv using the Sherman-Morrison formula
        self.sherman_morrison_update(v)

        # Add experience to replay buffer (convert to CPU if necessary)
        self.replay_buffer.add(context[action].to(torch.float32).cpu().numpy(), reward)

        self.T += 1
        self.train()

    def sherman_morrison_update(self, v):
        # Ensure v is a column vector
        if v.dim() == 1:
            v = v.unsqueeze(1)

        # Perform the Sherman-Morrison update on the device
        v = v.to(self.device)
        sigma_v = self.sigma_inv @ v  # Shape: [numel, 1]
        denominator = 1 + v.T @ sigma_v  # Shape: [1, 1]

        # Avoid division by zero
        denominator = denominator + 1e-6

        numerator = sigma_v @ sigma_v.T  # Shape: [numel, numel]
        self.sigma_inv -= numerator / denominator

    def train(self):
        if self.T > self.K and self.T % 1 == 0:
            for _ in range(2):
                x, y = self.replay_buffer.sample(64)
                x = torch.tensor(x, dtype=dtype_cuda, device=self.device)
                y = torch.tensor(y, dtype=dtype_cuda, device=self.device).view(-1, 1)

                y_hat = self.net(x)
                loss = F.mse_loss(y_hat, y)

                # Regularization term
                theta = torch.cat(
                    [w.flatten() for w in self.net.parameters() if w.requires_grad]
                ).to(self.device)
                loss += self.reg * torch.norm(theta - self.theta0) ** 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

