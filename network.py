import torch
import torch.nn as nn

'''
This file contains the neural network classes for the bandit modules and the aggregation network, which are used in the MuFasa algorithm, 
see the paper "Multi-facet Contextual Bandits: A Neural Network Perspective", original code at https://github.com/banyikun/KDD2021_MuFasa 
'''

class ModuleNetwork(nn.Module):
    '''
    This class defines the neural network for each bandit module, which takes in a context and outputs the expected reward for each arm.
    '''
    def __init__(self, idx, name, input_dim, hidden_size, num_arms):
        super(ModuleNetwork, self).__init__()
        self.__index__ = idx
        self.__function__ = name

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.arm_embeddings = nn.Embedding(num_arms, hidden_size)
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        # Compute the dot product of the context and the arm embeddings
        out = torch.matmul(out, self.arm_embeddings.weight.t())
        return out

class AggregateNetwork(nn.Module):
    '''
    This class defines the neural network that aggregates the outputs of the bandit modules to compute the expected reward for each combination of arms.
    '''
    def __init__(self, module_networks, hidden_size):
        super(AggregateNetwork, self).__init__()
        self.module_networks = nn.ModuleList(module_networks)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(len(module_networks), hidden_size)  # Transform from num_bandits to hidden_size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Flatten back to 1 value per combination

    def forward(self, x):
        # Assume x is a list of contexts, each corresponding to a bandit network
        outputs = [net(x).squeeze(0) for net in self.module_networks]  # Process each context through its bandit and remove batch dimension
        combined = torch.cartesian_prod(*outputs)  # Compute the Cartesian product of outputs
        combined = combined.view(1, -1, len(self.module_networks))  # Reshape to [1, num_combinations, num_bandits]
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)
        return out.view(-1)  # Flatten to [num_combinations]



class LSTMModuleNetwork(nn.Module):
    '''
    This class defines the neural network for each bandit module, which uses an LSTM to model the sequence of actions and rewards.
    '''
    def __init__(self, input_size, hidden_size, num_arms, lambd=1.0, nu=0.1):
        super(ModuleNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size + 1, hidden_size)  # +1 for the subreward input
        self.fc_rewards = nn.Linear(hidden_size, num_arms)  # Output a reward for each arm
        self.hidden_size = hidden_size
        self.num_arms = num_arms
        self.lambd = lambd
        self.nu = nu

        # Arm-level
        # self.U = lambd * torch.ones((self.num_arms,))  # Confidence parameter

        # Parameter-level
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.U = lambd * torch.ones((self.total_param,))

    def forward(self, input, previous_reward, hidden):
        # Concatenate previous reward with the input embedding
        combined_input = torch.cat([input, previous_reward.squeeze(-1)], dim=-1)
        lstm_out, hidden = self.lstm(combined_input.view(1, 1, -1), hidden)
        arm_rewards = self.fc_rewards(lstm_out.view(1, -1))

        # Compute softmax probabilities (for informational purposes)
        arm_probs = torch.softmax(arm_rewards, dim=1)

        # Intialize gradients and calculate the UCB for each arm
        ucb_values = []
        grads_list = []
        for i in range(self.num_arms):
            self.lstm.zero_grad()
            reward_i = arm_rewards[0, i]
            reward_i.backward(retain_graph=True)

            # Get gradients for all parameters and flatten them
            grads = torch.cat([p.grad.flatten().detach() for p in self.parameters() if p.grad is not None])
            # sigma2 = self.lambd * torch.sum(grads ** 2) / self.U[i]
            sigma2 = self.lambd * grads * grads / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            # Compute the UCB value
            ucb = self.nu * sigma.item()
            ucb_values.append(ucb)
            grads_list.append(grads)
        print(f"UCB values: {ucb_values}")
        selected_arm = torch.tensor([torch.argmax(arm_rewards.squeeze(0) +  torch.tensor(ucb_values))], dtype=torch.long)
        selected_reward = arm_rewards.gather(1, selected_arm.unsqueeze(1))

        # Arm-level
        # self.U[selected_arm] += torch.sum(grads ** 2) # Update U parameter for the selected arm

        # Parameter-level
        self.U += grads_list[selected_arm] ** 2


        return selected_reward, arm_probs, selected_arm, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

class LSTMAggregateNetwork(nn.Module):
    '''
    This class defines the neural network that aggregates the outputs of the bandit modules based on a tree structure.
    '''
    def __init__(self, module_dict, input_size):
        super(AggregateNetwork, self).__init__()
        self.module_dict = module_dict

    def forward(self, tree, program_input):
        outputs = {}
        reward_memory = {}
        hidden_states = {i: self.module_dict[node[0]].init_hidden() for i, node in enumerate(tree)}
        
        for i, node in enumerate(tree):
            module_function, children = node
            module = self.module_dict[module_function]
            module.__function__ = module_function
            input_tensor = program_input
            print(f"Current node {i}: {module_function}")
            if children:
                # Use the rewards of children as part of the input
                child_rewards = torch.cat([reward_memory[child] for child in children])
                aggregated_rewards = torch.mean(child_rewards, dim=0).unsqueeze(0)

                for name, reward in [(outputs[child][0], reward_memory[child]) for child in children]:
                    print(f"Reward from {name}: {reward}")
            else:
                aggregated_rewards = torch.zeros(1, 1) # Placeholder for initial subreward

            print(f"Previous reward: {aggregated_rewards}")
            selected_reward, arm_probs, selected_arm, hidden_states[i] = module(
                input_tensor, aggregated_rewards, hidden_states[i])
            outputs[i] = (module.__function__, selected_reward, arm_probs, selected_arm)
            reward_memory[i] = selected_reward

            print(f"Selected reward: {selected_reward}")
            print("**********")
        return reward_memory, outputs

    def init_modules(self):
        for module in self.module_dict.values():
            module.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)