import numpy as np
from operator import mul
from functools import reduce
from bandit import ContextualBandit, Reinforce, StructuredReinforce

class Router():
    def __init__(self, routing_info, routing_options, cost_weighting=0):
        self.routing_info = {key: len(routing_options[key.split("(")[0]]) for key in routing_info.keys()}
        self.num_arms = reduce(mul, [v for v in self.routing_info.values()])
        self.arm_costs = self.compute_costs(self.num_arms, routing_info, routing_options)
        self.lamb = 0.0001 # Regularization parameter
        self.nu = 0.1 # Exploration-exploitation trade-off parameter
        self.input_size = 3 # Size of the input features
        self.hidden_size = 128 # Hidden size of the neural network
        self.cost_weighting = cost_weighting # Cost weighting parameter
        print("Cost weighting: ", self.cost_weighting)
        self.initialize()
        self.t = 0
    
    def initialize(self):
        '''
        This function initializes the routing algorithm.
        '''
        # self.algo = ContextualBandit(self.num_arms, self.arm_costs, self.cost_weighting, self.lamb, self.nu)
        self.algo = Reinforce(self.num_arms, self.arm_costs, self.cost_weighting, self.lamb, self.nu)

    def compute_costs(self, num_arms, routing_info, routing_options):
        '''
        This function computes the costs for each arm based on the routing information.

        Args:
            num_arms: Number of arms
            routing_info: Routing information
            routing_options: Routing options

        Returns:
            arm_costs: List of costs for each arm

        '''
        arm_costs = []
        for i in range(num_arms):
            arm_indices = self.idx_to_arms(i)
            cost = 0
            for j, key in enumerate(routing_info.keys()):
                cost += routing_options[key.split("(")[0]][arm_indices[j]]
            arm_costs.append(cost)
        return arm_costs

    def idx_to_arms(self, arm_idx):
        '''
        This function converts the arm index to a list of arm indices based on the routing info.
        For example, if routing_info = {'find("dog")': 2, 'exists("cat")': 2}, then arm_idx = 3 corresponds to ['find("dog")': 1, 'exists("cat")': 1], arm_idx = 2 corresponds to ['find("dog")': 0, 'exists("cat")': 1]

        Args:
            arm_idx: Index of the arm

        Returns:
            arm_indices: List of arm indices based on the routing info

        '''
        if arm_idx >= self.num_arms:
            raise ValueError("Invalid arm index, valid range: [0, {}]".format(self.num_arms-1))
        arm_indices = []
        for key, value in self.routing_info.items():
            arm_value = arm_idx % value
            assert arm_value < value, "Invalid arm value"
            arm_indices.append(arm_value)
            arm_idx = arm_idx // value
        return arm_indices

    def arms_to_idx(self, arm_indices):
        '''
        This function converts the list of arm indices to an arm index based on the routing info.
        For example, if routing_info = {'find("dog")': 2, 'exists("cat")': 2}, then ['find("dog")': 1, 'exists("cat")': 1] corresponds to arm_idx = 3, ['find("dog")': 0, 'exists("cat")': 1] corresponds to arm_idx = 2

        Args:
            arm_indices: List of arm indices

        Returns:
            arm_idx: Index of the arm

        '''
        if len(arm_indices) != len(self.routing_info):
            raise ValueError("Invalid arm indices")
        if any(arm_indices[i] >= list(self.routing_info.values())[i] for i in range(len(arm_indices))):
            raise ValueError("Invalid arm indices")

        arm_idx = 0
        multiplier = 1
        for i, value in enumerate(self.routing_info.values()):
            arm_idx += arm_indices[i] * multiplier
            multiplier *= value
        return arm_idx
    
    def select(self, context):
        '''
        This function selects the arm based on the input context.

        Args:
            context: Input context

        Returns:
            selected_arms: Dictionary containing the selected arms for each routing option
            arm_idx: Index of the arm selected

        '''
        input_feature = np.array(context)
        arm_idx, _, _ = self.algo.select(input_feature, self.t) 
        if arm_idx is None:
            raise ValueError("Invalid arm index")
        selected_arms = self.idx_to_arms(arm_idx)
        return {key: selected_arms[i] for i, key in enumerate(self.routing_info.keys())}, arm_idx
    
    def update(self, context, arm_idx, reward):
        '''
        This function updates the routing algorithm based on the reward received.

        Args:
            context: Input context
            arm_idx: Index of the arm
            reward: Reward received

        '''
        self.t += 1
        self.algo.update(np.array(context), arm_idx, reward, self.t)
        if self.t % 32 == 0:
            self.algo.train()

class StructuredRouter():
    def __init__(self, routing_info, routing_options, cost_weighting=0):
        self.routing_info = {key: routing_options[key.split("(")[0]] for key in routing_info.keys()} 
        self.lamb = 0.0001 # Regularization parameter
        self.nu = 0.1 # Exploration-exploitation trade-off parameter
        self.input_size = 3 # Size of the input features
        self.hidden_size = 128 # Hidden size of the neural network
        self.cost_weighting = cost_weighting # Cost weighting parameter
        print("Cost weighting: ", self.cost_weighting)
        self.initialize()
        self.t = 0
    
    def initialize(self):
        self.algo = StructuredReinforce(self.routing_info, self.cost_weighting, self.lamb, self.nu)
    
    def select(self, context):
        '''
        This function selects the arm based on the input context.

        Args:
            context: Input context

        Returns:
            selected_arms: Dictionary containing the selected arms for each routing option
            
        '''
        input_feature = np.array(context)
        selected_arms, _, _ = self.algo.select(input_feature, self.t) 
        return {key: selected_arms[i] for i, key in enumerate(self.routing_info.keys())}, selected_arms
    
    def update(self, context, arm_indices, reward):
        '''
        This function updates the routing algorithm based on the reward received.

        Args:
            context: Input context
            arm_idx: Index of the arm
            reward: Reward received
            
        '''
        self.t += 1
        self.algo.update(np.array(context), arm_indices, reward, self.t)
        if self.t % 32 == 0:
            self.algo.train()