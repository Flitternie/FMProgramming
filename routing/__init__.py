import numpy as np
from operator import mul
from functools import reduce
from typing import Dict, List, Tuple, Union, Optional

from routing.algorithms import (
    ContextualBandit,
    Reinforce,
    StructuredReinforce,
    ParallizedStructuredReinforce
)

"""
This file contains the routing system interface that connects the decision-making algorithms (see algorithms.py).
"""

# ---------------------------------------------------------------------------- #
#                                   Router                                     #
# ---------------------------------------------------------------------------- #

class Router:
    """
    Router for decision-making between different routing strategies.

    Supports contextual bandits and reinforcement learning strategies
    to select the optimal routing configuration given an input context.
    """

    def __init__(
        self,
        routing_info: Dict[str, int],
        routing_options: Dict[str, List[float]],
        cost_weighting: float = 0,
        config: Optional[str] = None,
        update_freq: int = 64,
        num_epochs: int = 5,
        batch_size: int = 16
    ):
        """
        Args:
            routing_info (dict): Information about routing decisions and options.
            routing_options (dict): Mapping of routing options with associated costs.
            cost_weighting (float): Weight assigned to cost in reward calculations.
            config (str): Routing strategy: 'bandit' or 'reinforce'.
            update_freq (int): Frequency of updates to the algorithm.
            num_epochs (int): Number of epochs for training the model.
            batch_size (int): Size of batches for training.
        """
        self.routing_info = {
            key: len(routing_options[key.split("(")[0]])
            for key in routing_info.keys()
        }

        self.num_arms = reduce(mul, self.routing_info.values())
        self.arm_costs = self.compute_costs(self.num_arms, routing_info, routing_options)

        self.lamb = 0.0001
        self.nu = 0.1
        self.input_size = 3
        self.hidden_size = 128
        self.update_freq = update_freq 
        self.num_epochs = num_epochs 
        self.batch_size = batch_size 
        self.cost_weighting = cost_weighting

        print("Cost weighting:", self.cost_weighting)

        self.initialize(config)
        self.t = 0

    def initialize(self, config: Optional[str]) -> None:
        """Initialize the routing algorithm based on the configuration."""
        if config == "bandit":
            self.algo = ContextualBandit(
                self.num_arms, self.arm_costs, self.cost_weighting, self.lamb, self.nu
            )
        elif config == "reinforce":
            self.algo = Reinforce(
                self.num_arms, self.arm_costs, self.cost_weighting, self.lamb, self.nu
            )
        else:
            raise ValueError("Invalid configuration. Valid options: 'bandit', 'reinforce'.")

    def compute_costs(
        self,
        num_arms: int,
        routing_info: Dict[str, int],
        routing_options: Dict[str, List[float]]
    ) -> List[float]:
        """Compute costs for all possible arm combinations."""
        arm_costs = []
        for i in range(num_arms):
            arm_indices = self.idx_to_arms(i)
            cost = 0
            for j, key in enumerate(routing_info.keys()):
                option_key = key.split("(")[0]
                cost += routing_options[option_key][arm_indices[j]]
            arm_costs.append(cost)
        return arm_costs

    def idx_to_arms(self, arm_idx: int) -> List[int]:
        """
        Convert an arm index into a list of arm decisions.

        Returns:
            list: Arm decisions corresponding to each routing key.
        """
        if arm_idx >= self.num_arms:
            raise ValueError(f"Invalid arm index. Valid range: [0, {self.num_arms - 1}]")

        arm_indices = []
        for key, value in self.routing_info.items():
            arm_value = arm_idx % value
            arm_indices.append(arm_value)
            arm_idx //= value

        return arm_indices

    def arms_to_idx(self, arm_indices: List[int]) -> int:
        """
        Convert a list of arm decisions into a single arm index.

        Returns:
            int: Arm index corresponding to the given arm decisions.
        """
        if len(arm_indices) != len(self.routing_info):
            raise ValueError("Invalid arm indices length.")

        if any(arm_indices[i] >= list(self.routing_info.values())[i] for i in range(len(arm_indices))):
            raise ValueError("Arm index out of range for routing options.")

        arm_idx = 0
        multiplier = 1
        for i, value in enumerate(self.routing_info.values()):
            arm_idx += arm_indices[i] * multiplier
            multiplier *= value

        return arm_idx

    def select(self, context: Union[np.ndarray, List[float]]) -> Tuple[Dict[str, int], int]:
        """
        Select the best routing arm based on context.

        Args:
            context (array or list): Input context features.

        Returns:
            tuple: (selected arms dict, arm index)
        """
        input_feature = np.array(context)
        arm_idx, _, _ = self.algo.select(input_feature, self.t)

        if arm_idx is None:
            raise ValueError("Invalid arm index selected.")

        selected_arms = self.idx_to_arms(arm_idx)

        return (
            {key: selected_arms[i] for i, key in enumerate(self.routing_info.keys())},
            arm_idx
        )

    def update(
        self,
        context: Union[np.ndarray, List[float]],
        arm_idx: int,
        reward: float,
        reward_mapping: Optional[Dict] = None
    ) -> None:
        """
        Update the routing algorithm with the reward from the last decision.

        Args:
            context (array or list): Input context features.
            arm_idx (int): Chosen arm index.
            reward (float): Observed reward.
        """
        self.t += 1
        self.algo.update(np.array(context), arm_idx, reward, self.t)

        if self.t % 64 == 0:
            self.algo.train(num_epochs=5, batch_size=16)

# ---------------------------------------------------------------------------- #
#                             Structured Router                                #
# ---------------------------------------------------------------------------- #

class StructuredRouter:
    """
    StructuredRouter implements a structured reinforcement learning-based router.

    This router makes separate routing decisions for each routing function and supports
    structured exploration and exploitation.
    """

    def __init__(
        self,
        routing_info: Dict[str, int],
        routing_options: Dict[str, List[float]],
        cost_weighting: float = 0,
        update_freq: int = 64,
        num_epochs: int = 5,
        batch_size: int = 16
    ):
        """
        Args:
            routing_info (dict): Mapping of routing decisions.
            routing_options (dict): Available options and associated costs.
            cost_weighting (float): Cost weighting factor in optimization.
            update_freq (int): Frequency of updates to the algorithm.
            num_epochs (int): Number of epochs for training the model.
            batch_size (int): Size of batches for training.
        """
        self.routing_info = {
            key: routing_options[key.split("(")[0]]
            for key in routing_info.keys()
        }

        self.num_arms = reduce(mul, [len(v) for v in self.routing_info.values()])

        self.lamb = 0.0001
        self.nu = 0.1
        self.input_size = 3
        self.hidden_size = 128
        self.update_freq = update_freq
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cost_weighting = cost_weighting

        print("Cost weighting:", self.cost_weighting)

        self.initialize()
        self.t = 0

    def initialize(self) -> None:
        """Initialize the structured reinforcement learning algorithm."""
        self.algo = ParallizedStructuredReinforce(
            self.routing_info, self.cost_weighting, self.lamb, self.nu
        )

    def select(self, context: Union[np.ndarray, List[float]]) -> Tuple[Dict[str, int], List[int]]:
        """
        Select routing decisions based on context.

        Args:
            context (array or list): Input context features.

        Returns:
            tuple: (selected arms dict, list of selected arms)
        """
        input_feature = np.array(context)
        selected_arms, _, _ = self.algo.select(input_feature, self.t)

        return (
            {key: selected_arms[i] for i, key in enumerate(self.routing_info.keys())},
            selected_arms
        )

    def update(
        self,
        context: Union[np.ndarray, List[float]],
        arm_indices: List[int],
        reward: float,
        reward_mapping: Optional[Dict] = None
    ) -> None:
        """
        Update the structured routing algorithm with new reward feedback.

        Args:
            context (array or list): Input context features.
            arm_indices (list): List of selected arms.
            reward (float): Reward received.
            reward_mapping (dict): Additional reward mappings.
        """
        self.t += 1

        self.algo.update(np.array(context), arm_indices, reward, reward_mapping, self.t)

        if self.t % 64 == 0:
            self.algo.train(num_epochs=5, batch_size=16)
