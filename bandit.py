import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch import device
import tqdm
from scipy.optimize import linprog
from itertools import product
from network import ModuleNetwork, AggregateNetwork, CNN
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from concurrent.futures import ThreadPoolExecutor, as_completed

'''
This file contains the implementation of the following bandit algorithms:
1. Multi-facet Contextual Bandits (MuFasa)
2. Upper Confidence Bound (UCB1)
3. UCB with Linear Programming (UCB_ALP)
4. Contextual Bandits (ContextualBandit)
'''


# Multi-facet Contextual Bandit algorithm, from paper "Multi-facet Contextual Bandits: A Neural Network Perspective", original code at https://github.com/banyikun/KDD2021_MuFasa
class MuFasa:
    def __init__(self, modules, input_size, lambd=1, nu=0.1, hidden_size=128):
        # Initialize the network
        self.input_size = input_size  # Dimension of input context
        self.modules = []
        for idx, (name, num_arms) in enumerate(modules.items()):
            self.modules.append(ModuleNetwork(idx, name, input_size, hidden_size, num_arms).to(device))
        self.aggregator = AggregateNetwork(self.modules, input_size).to(device)
        
        self.context_list = []  # Store contexts
        self.reward_list = []  # Store rewards
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
        return arm
    
    def update(self, context, _, final_r, t):
        # Update context and reward lists with observed final rewards
        self.context_list.append(context)
        self.reward_list.append(final_r)

    def train(self):
        # Training the network with stored contexts and rewards
        context_list = self.context_list
        reward = self.reward_list
                
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


# Classical UCB algorithm
class UCB1:
    def __init__(self, num_arms, alpha=1):
        self.num_arms = num_arms
        self.counts = np.zeros(self.num_arms)  # Number of times each arm was pulled
        self.values = np.zeros(self.num_arms)  # Average reward for each arm
        self.alpha = alpha  # Exploration hyperparameter

    def select(self, context, t):
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


# UCB algorithm with linear programming for budget constraints
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

    def select(self, context, t):
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


class ContextualBandit:
    def __init__(self, num_arms, arm_costs, cost_weighting, lambd=1, nu=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_arms = num_arms
        self.arm_costs = arm_costs
        self.cost_weighting = cost_weighting
        self.model = CNN(num_arms).to(self.device)
        self.lambd = lambd
        self.nu = nu
        self.lr = 0.001

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = lambd * torch.ones((total_params,)).to(next(self.model.parameters()).device)

        self.optimizer = optim.AdamW(list(self.model.parameters()), lr=self.lr)
        self._initialize_cache()

    
    def _initialize_cache(self):
        # Storage for learning
        self.context_list = []
        self.arm_list = []
        self.reward_list = []
    
    def select(self, context, t):
        # Convert context to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device)
        context_tensor = context_tensor.unsqueeze(0)  # Add batch dimension
        model_outputs = self.model(context_tensor).squeeze(0)
        
        g_list = []  # Gradients list
        sampled_rewards = []  # Sampled rewards
        true_rewards = []  # True rewards
        
        # Compute the sigma and sampled reward for each arm combination using gradient-based Thompson Sampling
        for arm_index in range(self.num_arms):
            reward_prediction = model_outputs[arm_index]

            # Calculate gradient for Thompson Sampling
            self.model.zero_grad()
            reward_prediction.backward(retain_graph=True)

            # Collect gradients
            gradients = torch.cat([p.grad.flatten().detach() for p in list(self.model.parameters())])
            
            # Sample from Normal distribution based on gradient norms for Thompson Sampling
            sigma = torch.sqrt(self.lambd * torch.sum(gradients * gradients / self.U))
            sampled_r = reward_prediction.item() + Normal(0, self.nu * sigma.item()).sample().item()
            sampled_r = sampled_r - self.arm_costs[arm_index] * self.cost_weighting
            true_rewards.append(reward_prediction.item())
            sampled_rewards.append(sampled_r)
            g_list.append(gradients)
        
        # Select arm based on Thompson Sampling
        selected_arm = np.argmax(sampled_rewards)
        self.U += g_list[selected_arm] * g_list[selected_arm]
        return selected_arm, sampled_rewards, true_rewards
    
    def update(self, context, arm_idx, final_r, t):
        # Update context and reward lists with observed final rewards
        self.context_list.append(context)
        self.arm_list.append(arm_idx)
        self.reward_list.append(final_r)
    
    def train(self, num_epochs=5, batch_size=8):
         # Train the network on observed data
        index = np.arange(len(self.reward_list))
        context_list = self.context_list
        arm_list = self.arm_list
        reward_list = self.reward_list
    
        np.random.shuffle(index)
        length = len(reward_list)
        
        if length == 0:
            return 0.0  # No data to train on
        
        num_batches = max(1, length // batch_size)
        total_loss = 0
        
        for epoch in range(num_epochs):
            np.random.shuffle(index)
            for batch_idx in range(num_batches):
                batch_loss = 0
                self.optimizer.zero_grad()
                
                # Process each item in the batch
                for i in range(batch_size):
                    idx = index[min(batch_idx * batch_size + i, length - 1)]
                    context = context_list[idx]
                    context = torch.tensor(np.array(context), dtype=torch.float32, device=self.device)
                    if context.dim() == 3:
                        context = context.unsqueeze(0)
                    model_outputs = self.model(context)
                    r = reward_list[idx]
                    arm = arm_list[idx]
        
                    predicted_reward = model_outputs.squeeze(0)[arm]
                    
                    predicted_reward = predicted_reward - self.arm_costs[arm] * self.cost_weighting

                    # MSE loss for reward prediction
                    loss = (predicted_reward - r) ** 2
                    batch_loss += loss
                
                # Average batch loss
                batch_loss /= batch_size
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss.item()
        
        average_loss = total_loss / (num_batches * num_epochs)
        self._initialize_cache()
        return average_loss


class Reinforce:
    def __init__(self, num_arms, arm_costs, cost_weighting, lambd, nu, lr=0.001, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_arms = num_arms
        self.arm_costs = torch.tensor(arm_costs, device=self.device)
        self.cost_weighting = cost_weighting
        self.lambd = lambd
        self.nu = nu
        self.lr = lr

        # Model Initialization
        self.model = CNN(num_arms).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Initialize U (parameter uncertainties) and cache
        self._initialize_uncertainties()
        self._initialize_cache()

    def _initialize_uncertainties(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = self.lambd * torch.ones(total_params, device=self.device)

    def _initialize_cache(self):
        self.context_list = []
        self.arm_list = []
        self.reward_list = []

    def select(self, context, t):
        # Convert context to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device)
        context_tensor = context_tensor.unsqueeze(0)  # Add batch dimension
        model_outputs = self.model(context_tensor).squeeze(0)

        # Store gradients, rewards, and sampled rewards
        g_list = []
        sampled_rewards = torch.zeros(self.num_arms, device=self.device)
        true_rewards = model_outputs.clone().detach()  # Store predicted rewards

        for arm_index in range(self.num_arms):
            # Zero gradients and compute gradient for the arm
            self.model.zero_grad()
            reward_prediction = model_outputs[arm_index]
            reward_prediction.backward(retain_graph=True)

            # Collect gradients
            gradients = torch.cat([p.grad.flatten() for p in self.model.parameters()])
            
            # Compute sigma (uncertainty measure)
            sigma = torch.sqrt(torch.clamp(self.lambd * torch.sum(gradients * gradients / self.U), min=1e-6))
            
            # Sample reward with added Gaussian noise (Thompson Sampling)
            sampled_r = reward_prediction.item() + Normal(0, self.nu * sigma.item()).sample().item()
            # Compute the local costs for the current arm selection
            sampled_r -= self.arm_costs[arm_index] * self.cost_weighting
            sampled_rewards[arm_index] = sampled_r
            g_list.append(gradients)

        self.g_list = g_list  # Store gradients for training
        # Select the arm with the highest sampled reward
        selected_arm = torch.argmax(sampled_rewards).item()

        return selected_arm, sampled_rewards.tolist(), true_rewards.tolist()

    def update(self, context, arm_idx, final_r, t):
        # Cache training data for later use
        self.context_list.append(context)
        self.arm_list.append(arm_idx)
        self.reward_list.append(final_r)

        self.U += self.g_list[arm_idx] ** 2  # Update U using the gradients of selected arm
        self.g_list = None # Clear gradients after updating U


    def train(self, num_epochs, batch_size):
        assert len(self.context_list) == len(self.arm_list) == len(self.reward_list), "Training data mismatch"
        if len(self.reward_list) == 0:
            return -1

        self.model.train()  # Set model to training mode
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = len(self.reward_list) // batch_size + 1
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(self.reward_list))
                
                if start_idx >= end_idx:
                    continue
                
                # Fetch batch data
                # Creating a tensor from a list of numpy.ndarrays is extremely slow. Converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
                batch_contexts = torch.tensor(np.array(self.context_list[start_idx:end_idx]), dtype=torch.float32, device=self.device)
                batch_arm_indices = torch.tensor(np.array(self.arm_list[start_idx:end_idx]), dtype=torch.long, device=self.device)
                batch_rewards = torch.tensor(np.array(self.reward_list[start_idx:end_idx]), dtype=torch.float32, device=self.device)
                
                # Forward pass
                logits = self.model(batch_contexts)
                probs = torch.softmax(logits, dim=1)
                
                # Compute batch loss (negative REINFORCE loss)
                log_probs = torch.log(probs[range(len(batch_arm_indices)), batch_arm_indices] + 1e-6)
                loss = -torch.mean(batch_rewards * log_probs)
                
                # Backward pass and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average loss: {avg_loss:.6f}")
        
        self._initialize_cache()  # Clear cache after training


class StructuredReinforce:
    def __init__(self, module_info, cost_weighting, lambd, nu, lr=0.001):
        self.num_modules = len(module_info.keys())
        self.policies = [
            Reinforce(
                num_arms=len(arm_costs),
                arm_costs=arm_costs,
                cost_weighting=cost_weighting,
                lambd=lambd,
                nu=nu
            )
            for i, arm_costs in enumerate(module_info.values())
        ]

    def select(self, context, t):
        # Select arm for each policy
        selected_arms = []
        sampled_rewards_list = []
        true_rewards_list = []
        for policy in self.policies:
            selected_arm, sampled_rewards, true_rewards = policy.select(context, t)
            selected_arms.append(selected_arm)
            sampled_rewards_list.append(sampled_rewards)
            true_rewards_list.append(true_rewards)
        return selected_arms, sampled_rewards_list, true_rewards_list
    
    def update(self, context, arm_indices, final_r, t):
        # Update context and reward lists with observed final rewards
        for i, policy in enumerate(self.policies):
            policy.update(context, arm_indices[i], final_r, t)
    
    def train(self, num_epochs, batch_size):
        # Train the network using REINFORCE algorithm
        for policy in self.policies:
            policy.train(num_epochs, batch_size)


class ParallizedStructuredReinforce:
    def __init__(self, module_info, cost_weighting, lambd, nu, lr=0.001):
        self.num_modules = len(module_info.keys())
        self.devices = [torch.device(f'cuda:{i % torch.cuda.device_count()}') for i in range(self.num_modules)]
        self.policies = [
            Reinforce(
                num_arms=len(arm_costs),
                arm_costs=arm_costs,
                cost_weighting=cost_weighting,
                lambd=lambd,
                nu=nu,
                device=self.devices[i]
            )
            for i, arm_costs in enumerate(module_info.values())
        ]

    def select(self, context, t):
        # Select arm for each policy
        selected_arms = []
        sampled_rewards_list = []
        true_rewards_list = []
        for policy in self.policies:
            selected_arm, sampled_rewards, true_rewards = policy.select(context, t)
            selected_arms.append(selected_arm)
            sampled_rewards_list.append(sampled_rewards)
            true_rewards_list.append(true_rewards)
        return selected_arms, sampled_rewards_list, true_rewards_list

    def update(self, context, arm_indices, final_r, reward_mapping, t):
        # Update context and reward lists with observed final rewards
        for i, policy in enumerate(self.policies):
            if reward_mapping[i] == 1:
                policy.update(context, arm_indices[i], final_r, t)

    def train(self, num_epochs, batch_size):
        # Parallelize training across multiple GPUs using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(policy.train, num_epochs, batch_size) 
                for policy in self.policies
            ]
            for future in as_completed(futures):
                try:
                    future.result()  # Ensure exceptions are caught
                except Exception as e:
                    print(f"Training failed for a policy: {e}")



class Supervised:
    def __init__(self, num_arms, input_size, hidden_size=512, device='cpu'):
        # Initialize the network
        self.input_size = input_size  # Dimension of input context
        self.num_arms = num_arms
        self.hidden_size = hidden_size
        self.device = device

        # Arm embeddings
        # self.arm_embeddings = nn.Embedding(self.num_arms, self.hidden_size).to(self.device)
        
        # Context transformation
        self.context_transform = nn.Linear(self.input_size, self.hidden_size).to(self.device)
        self.activation = nn.ReLU().to(self.device)
        
    def train(self, contexts, arm_indices, rewards, num_epochs=5, batch_size=16, lr=0.0003):
        # Prepare data for training
        contexts_tensor = torch.tensor(contexts, dtype=torch.float32).to(self.device)
        arm_indices_tensor = torch.tensor(arm_indices, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(contexts_tensor, arm_indices_tensor, rewards_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = optim.SGD(list(self.arm_embeddings.parameters()) + 
                              list(self.context_transform.parameters()), 
                              lr=lr)

        loss_fn = nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_contexts, batch_arms, batch_rewards in pbar:
                # Forward pass
                optimizer.zero_grad()
                transformed_contexts = self.activation(self.context_transform(batch_contexts))
                arm_embeddings = self.arm_embeddings(batch_arms)
                predicted_rewards = torch.sum(transformed_contexts * arm_embeddings, dim=1)

                # Compute loss
                loss = loss_fn(predicted_rewards, batch_rewards)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_contexts.size(0)
                pbar.set_postfix(loss=total_loss / len(data_loader.dataset))
        
        average_loss = total_loss / len(data_loader.dataset)
        return average_loss
    
    def select(self, context):
        # Convert context to tensors
        input_tensor = torch.tensor(context, dtype=torch.float32).to(self.device)
        transformed_context = self.activation(self.context_transform(input_tensor))
        
        # Predict reward for each arm
        arm_embeddings = self.arm_embeddings.weight  # (num_arms, hidden_size)
        reward_predictions = torch.matmul(transformed_context, arm_embeddings.T)  # (num_arms,)
        
        # Select the arm with the highest predicted reward
        return torch.argmax(reward_predictions).item()