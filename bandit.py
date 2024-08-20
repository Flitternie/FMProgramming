import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.optimize import linprog
from itertools import product
from network import ModuleNetwork, AggregateNetwork

# Check if CUDA (GPU) is available and set the device accordingly
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)


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
                
        optimizer = optim.SGD(self.aggregator.parameters(), lr=0.0001, weight_decay=self.lambd)
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
    def __init__(self, modules, alpha=2):
        n_arms = 1
        for _, num_arms in modules.items():
            n_arms *= num_arms
        self.num_arms = n_arms
        self.counts = np.zeros(self.num_arms)  # Number of times each arm was pulled
        self.values = np.zeros(self.num_arms)  # Average reward for each arm
        self.alpha = alpha  # Exploration hyperparameter

    def select(self, t):
        n_pulls = sum(self.counts)
        if 0 in self.counts:
            # Pull each arm at least once
            return np.argmin(self.counts)
        # Adjust exploration term using alpha
        ucb_values = self.values + np.sqrt(self.alpha * np.log(n_pulls) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the average reward using incremental formula
        self.values[chosen_arm] = value + (reward - value) / n


class UCB_ALP:
    def __init__(self, modules, module_costs, budget, horizon):
        self.modules = modules
        self.module_costs = [costs for costs in module_costs.values()]
        self.budget = budget
        self.horizon = horizon

        self.arms = self.generate_arms()
        self.num_arms = len(self.arms)
        self.costs = self.compute_arm_costs()

        self.ucbs = np.ones(self.num_arms)
        self.counts = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.remaining_budget = budget

    def generate_arms(self):
        """ Generate all possible combinations of modules filling the holes. """
        indices = [range(self.modules[func]) for func in self.modules.keys()]
        return list(product(*indices))
    
    def compute_arm_costs(self):
        """ Compute the cost of a given arm. """
        return np.array([sum(self.module_costs[hole][module] for hole, module in enumerate(arm)) for arm in self.arms])

    def update_ucb(self, t):
        """ Update the UCB values based on counts and observed rewards. """
        for i in range(self.num_arms):
            if self.counts[i] > 0:
                avg_reward = self.rewards[i] / self.counts[i]
                self.ucbs[i] = avg_reward + np.sqrt((2 * np.log(t)) / self.counts[i])
            else:
                self.ucbs[i] = 1  # High UCB to ensure exploration
    
    def solve_lp(self):
        """ Solve the linear programming problem to determine action probabilities. """
        c = -self.ucbs.flatten()  # Minimize the negative of UCBs to maximize UCBs
        A = [self.costs]  # Cost constraints
        b = [self.remaining_budget / self.horizon]  # Remaining budget per unit time
        bounds = [(0, 1)] * self.num_arms
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return np.zeros(self.num_arms)

    def select(self, t):
        self.update_ucb(t)
        if self.remaining_budget > 0:
            action_probs = self.solve_lp()
            if not np.any(action_probs):  # Check if weights are all zeros
                return None  # No action taken if all weights are zero due to budget constraints
            total_probs = np.sum(action_probs)
            normalized_weights = action_probs / total_probs
            chosen_arm = np.random.choice(range(self.num_arms), p=normalized_weights)
            # cost = self.costs[chosen_arm]
            return chosen_arm
        else:
            return None

    def update(self, chosen_arm, reward):
        self.rewards[chosen_arm] += reward
        self.counts[chosen_arm] += 1
        cost = self.costs[chosen_arm]
        self.remaining_budget -= cost
        self.horizon -= 1 

class UCBContextualBandit:
    def __init__(self, modules, input_size, alpha=1.0):
        self.alpha = alpha  # Exploration coefficient
        self.input_size = input_size  # Dimension of input context
        self.n_arms = 1
        for _, num_arms in modules.items():
            self.n_arms *= num_arms  # Total number of arm combinations
        
        self.counts = np.zeros(self.n_arms)  # Times each arm has been selected
        self.values = np.zeros(self.n_arms)  # Average reward of each arm
        self.modules = list(modules.keys())  # Names of the modules
        self.features = [torch.nn.Parameter(torch.randn(input_size)) for _ in range(self.n_arms)]  # Simulated features for each arm
        
        # Simulation of a simple linear predictor for contextual bandits
        self.weights = [torch.nn.Parameter(torch.randn(input_size)) for _ in range(self.n_arms)]  # Weights for each module

    def select(self, context):
        context_tensor = torch.tensor(context).float()
        ucb_values = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            # Simple linear prediction model
            predicted_reward = torch.dot(self.weights[i], context_tensor).item()
            
            if self.counts[i] == 0:
                # If an arm hasn't been pulled, prioritize it
                ucb_values[i] = float('inf')
            else:
                # Calculate UCB value for the arm
                exploration_term = np.sqrt(self.alpha * np.log(sum(self.counts) + 1) / self.counts[i])
                ucb_values[i] = predicted_reward + exploration_term
        
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

    def update(self, chosen_arm, reward, context):
        # Update the counts and the average reward values for the chosen arm
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = value + (reward - value) / n
        self.values[chosen_arm] = new_value
        
        # Simulate feature update (notional, for illustrative purposes)
        context_tensor = torch.tensor(context).float()
        gradient = (reward - torch.dot(self.weights[chosen_arm], context_tensor)) * context_tensor
        self.weights[chosen_arm] += 0.01 * gradient  # Fake learning rate

    def train(self):
        # Not needed for this simple implementation, unless real learning is to be done
        pass
