import numpy as np
from operator import mul
from functools import reduce
from bandit import MuFasa, UCB1, UCB_ALP, ContextualBandit, Reinforce

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
        self.algo = ContextualBandit(self.num_arms, self.arm_costs, self.cost_weighting, self.lamb, self.nu)
        # self.algo = Reinforce(self.num_arms, self.arm_costs, self.cost_weighting, self.lamb, self.nu)

    def compute_costs(self, num_arms, routing_info, routing_options):
        arm_costs = []
        for i in range(num_arms):
            arm_indices = self.idx_to_arms(i)
            cost = 0
            for j, key in enumerate(routing_info.keys()):
                cost += routing_options[key.split("(")[0]][arm_indices[j]]
            arm_costs.append(cost)
        return arm_costs

    def idx_to_arms(self, arm_idx):
        # Convert the arm index to a list of arm indices based on the routing info
        # For example, if routing_info = {'find("dog")': 2, 'exists("cat")': 2}, then arm_idx = 3 corresponds to ['find("dog")': 1, 'exists("cat")': 1], arm_idx = 2 corresponds to ['find("dog")': 0, 'exists("cat")': 1]
        arm_indices = []
        for key, value in self.routing_info.items():
            arm_value = arm_idx % value
            assert arm_value < value, "Invalid arm value"
            arm_indices.append(arm_value)
            arm_idx = arm_idx // value
        return arm_indices
    
    def select(self, context):
        '''
        This function selects the arm based on the Contextual Bandit algorithm.
        '''
        input_feature = np.array(context)
        arm_idx, _, _ = self.algo.select(input_feature, self.t) 
        if arm_idx is None:
            raise ValueError("Invalid arm index")
        selected_arms = self.idx_to_arms(arm_idx)
        return {key: selected_arms[i] for i, key in enumerate(self.routing_info.keys())}, arm_idx
    
    def update(self, context, arm_idx, reward):
        self.t += 1
        self.algo.update(np.array(context), arm_idx, reward, self.t)
        if self.t % 32 == 0:
            self.algo.train()
