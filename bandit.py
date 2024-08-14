import numpy as np
import torch
import torch.optim as optim
from network import ModuleNetwork, AggregateNetwork

# Check if CUDA (GPU) is available and set the device accordingly
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)
print(device)

# Define the MuFasa class
class MuFasa:
    def __init__(self, modules, input_size, lambd=1, nu=0.1, hidden_size=128):
        # Initialize the network
        self.input_size = input_size  # Dimension of input context
        self.modules = []
        for idx, (name, num_arms) in enumerate(modules.items()):
            self.modules.append(ModuleNetwork(idx, name, input_size, hidden_size, num_arms).to(device))
        self.aggregator = AggregateNetwork(self.modules, input_size).to(device)
        
        self.context_list = []  # Store contexts
        self.reward = []  # Store rewards
        self.lambd = lambd  # Regularization parameter
        self.total_param = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)  # Total number of parameters
        self.U = lambd * torch.ones((self.total_param,)).to(device)  # Confidence parameter
        self.nu = nu  # Exploration-exploitation trade-off parameter
        

    def select(self, context, t):
        # Convert context to tensors
        input = torch.tensor(context).float().to(device)
        
        def _factT(m):
            return np.sqrt(1 / (1 + m))
        
        mu = self.aggregator(input)  # Predict final reward
        g_list = []  # Gradients list
        sampled = []  # Sampled rewards
        avg_sigma = 0  # Average sigma
        avg_reward = 0  # Average reward
        
        # Compute the sigma and sampled reward for each arm combination
        for fx in mu:
            self.aggregator.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([para.grad.flatten().detach() for para in self.aggregator.parameters()])
            sigma2 = self.lambd * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + self.nu * sigma.item()
            sampled.append(sample_r)
            avg_sigma += sigma.item()
            avg_reward += sample_r
            g_list.append(g)
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), avg_sigma, avg_reward
    
    def update(self, context, final_r, t):
        # Update context and reward lists with observed final rewards
        self.context_list.append(context)
        self.reward.append(final_r)

    def train(self):
        # Training the network with stored contexts and rewards
        context_list = self.context_list
        reward = self.reward
                
        optimizer = optim.SGD(self.aggregator.parameters(), lr=1e-2, weight_decay=self.lambd)
        length = len(reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        total_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                context = context_list[idx]
                context = torch.from_numpy(context).float().to(device)
                r = reward[idx]
                optimizer.zero_grad()
                delta = self.aggregator(context).max() - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return total_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

class UCB1:
    def __init__(self, modules):
        n_arms = 1
        for _, num_arms in modules.items():
            n_arms *= num_arms
        self.counts = np.zeros(n_arms)  # Number of times each arm was pulled
        self.values = np.zeros(n_arms)  # Average reward for each arm

    def select(self):
        n_pulls = sum(self.counts)
        if 0 in self.counts:
            # Pull each arm at least once
            return np.argmin(self.counts)
        ucb_values = self.values + np.sqrt(2 * np.log(n_pulls) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the average reward using incremental formula
        self.values[chosen_arm] = value + (reward - value) / n