import numpy as np
import torch
import torch.optim as optim
from torch import device
from scipy.optimize import linprog
from itertools import product
from routing.networks import ModuleNetwork, AggregateNetwork, CNN
from torch.distributions import Normal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Union

"""
This file contains implementations of various multi-armed bandit algorithms and REINFORCE algorithm 
used for adaptive routing in FM Programming system. These algorithms help balance the trade-off
between computational cost and model performance by selecting the most appropriate
FM backend (or "arm") during inference.
"""

# ---------------------------------------------------------------------------------------- #
#                               Multi-facet Contextual Bandit                              #
# ---------------------------------------------------------------------------------------- #

class MuFasa:
    """
    Multi-facet Contextual Bandit (MuFasa) implementation using neural networks to model module interactions.

    Args:
        modules (dict): Dictionary mapping module names to the number of arms.
        input_size (int): Dimension of the context vector.
        lambd (float): Regularization parameter.
        nu (float): Exploration-exploitation tradeoff coefficient.
        hidden_size (int): Hidden layer size in networks.
    """

    def __init__(self, modules, input_size, lambd=1, nu=0.1, hidden_size=128):
        self.input_size = input_size
        self.modules = [
            ModuleNetwork(idx, name, input_size, hidden_size, num_arms).to(device)
            for idx, (name, num_arms) in enumerate(modules.items())
        ]
        self.aggregator = AggregateNetwork(self.modules, input_size).to(device)

        self.context_list = []
        self.reward_list = []
        self.lambd = lambd
        self.total_param = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
        self.U = lambd * torch.ones((self.total_param,)).to(device)
        self.nu = nu

    def select(self, context: np.ndarray, t: int) -> int:
        """
        Select an arm given the input context using Thompson Sampling strategy.

        Args:
            context (np.ndarray): The input context vector.
            t (int): Current timestep.

        Returns:
            int: Index of the selected arm.
        """
        input_tensor = torch.tensor(context).float().to(device)
        mu = self.aggregator(input_tensor)

        g_list = []
        sampled = []

        for fx in mu:
            self.aggregator.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.aggregator.parameters()])
            sigma2 = self.lambd * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            sample_r = fx.item() + self.nu * sigma.item()
            sampled.append(sample_r)
            g_list.append(g)

        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm

    def update(self, context: np.ndarray, _: int, final_r: float, t: int) -> None:
        """
        Update internal storage with observed rewards.

        Args:
            context (np.ndarray): The context vector.
            _ (int): Ignored arm index.
            final_r (float): Observed reward.
            t (int): Current timestep.
        """
        self.context_list.append(context)
        self.reward_list.append(final_r)

    def train(self) -> float:
        """
        Train the aggregator network on observed (context, reward) pairs.

        Returns:
            float: Average training loss.
        """
        optimizer = optim.SGD(self.aggregator.parameters(), lr=0.0001, weight_decay=self.lambd)
        indices = np.arange(len(self.reward_list))
        np.random.shuffle(indices)

        total_loss = 0
        cnt = 0

        while True:
            batch_loss = 0
            for idx in indices:
                context = torch.from_numpy(self.context_list[idx]).float().to(device)
                r = self.reward_list[idx]
                optimizer.zero_grad()
                delta = self.aggregator(context).max() - r
                loss = delta * delta
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_loss += loss.item()
                cnt += 1

                if cnt >= 1000:
                    return total_loss / 1000

            if batch_loss / len(indices) <= 1e-3:
                return batch_loss / len(indices)

# ---------------------------------------------------------------------------------------- #
#                                 Classical UCB Bandit                                     #
# ---------------------------------------------------------------------------------------- #

class UCB1:
    """
    Classical Upper Confidence Bound (UCB1) Bandit Algorithm.

    Args:
        num_arms (int): Number of available actions/arms.
        alpha (float): Exploration factor.
    """

    def __init__(self, num_arms: int, alpha: float = 1.0):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.alpha = alpha

    def select(self, context: np.ndarray, t: int) -> int:
        """
        Select an arm based on the UCB strategy.

        Args:
            context (np.ndarray): Unused.
            t (int): Current timestep.

        Returns:
            int: Selected arm index.
        """
        if 0 in self.counts:
            return np.argmin(self.counts)

        n_pulls = sum(self.counts)
        ucb_values = self.values + np.sqrt(self.alpha * np.log(n_pulls) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Update reward statistics for the chosen arm.

        Args:
            chosen_arm (int): The selected arm.
            reward (float): The observed reward.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n

# ---------------------------------------------------------------------------------------- #
#                           UCB with Adaptive Linear Programming                           #
# ---------------------------------------------------------------------------------------- #

class UCB_ALP:
    """
    Upper Confidence Bound with Adaptive Linear Programming (UCB-ALP) for budget-constrained selection.

    Args:
        modules (Dict[str, int]): Mapping from module names to number of arms.
        module_costs (Dict[str, List[float]]): Mapping from module names to list of costs per arm.
        budget (float): Total budget available.
        horizon (int): Total number of timesteps.
    """

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

    def generate_arms(self) -> List[Tuple[int]]:
        """Generate all combinations of arm selections."""
        indices = [range(self.modules[func]) for func in self.modules.keys()]
        return list(product(*indices))

    def compute_arm_costs(self) -> np.ndarray:
        """Compute the cost associated with each arm combination."""
        return np.array([
            sum(self.module_costs[hole][module] for hole, module in enumerate(arm))
            for arm in self.arms
        ])

    def update_ucb(self, t: int) -> None:
        """Update UCB values using current reward statistics."""
        for i in range(self.num_arms):
            if self.counts[i] > 0:
                avg_reward = self.rewards[i] / self.counts[i]
                self.ucbs[i] = avg_reward + np.sqrt((2 * np.log(t)) / self.counts[i])
            else:
                self.ucbs[i] = 1  # Force exploration

    def solve_lp(self) -> np.ndarray:
        """Solve a linear program to distribute probabilities over arms."""
        c = -self.ucbs.flatten()
        A_eq = [self.costs]
        b_eq = [self.remaining_budget / self.horizon]
        bounds = [(0, 1)] * self.num_arms
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        return res.x if res.success else np.zeros(self.num_arms)

    def select(self, context: np.ndarray, t: int) -> Optional[int]:
        """
        Select an arm based on UCB and linear programming.

        Args:
            context (np.ndarray): Context vector (not used in this algorithm).
            t (int): Current timestep.

        Returns:
            Optional[int]: Selected arm index or None if budget is exhausted.
        """
        self.update_ucb(t)
        if self.remaining_budget <= 0:
            return None

        action_probs = self.solve_lp()
        if not np.any(action_probs):
            return None
        chosen_arm = np.random.choice(range(self.num_arms), p=action_probs / action_probs.sum())
        return chosen_arm

    def update(self, chosen_arm: int, reward: float) -> None:
        """
        Update the arm's reward statistics and adjust budget.

        Args:
            chosen_arm (int): Index of selected arm.
            reward (float): Observed reward.
        """
        self.rewards[chosen_arm] += reward
        self.counts[chosen_arm] += 1
        self.remaining_budget -= self.costs[chosen_arm]
        self.horizon -= 1

# ---------------------------------------------------------------------------- #
#                            ContextualBandit Class                            #
# ---------------------------------------------------------------------------- #

class ContextualBandit:
    """
    Contextual Bandit with Thompson Sampling and gradient-based reward modeling.
    """

    def __init__(self, num_arms: int, arm_costs: List[float], cost_weighting: float,
                 lambd: float = 1.0, nu: float = 0.1):
        """
        Initialize the Contextual Bandit.

        Args:
            num_arms (int): Number of arms.
            arm_costs (List[float]): Cost associated with each arm.
            cost_weighting (float): Weight factor for cost in reward.
            lambd (float): Regularization parameter for uncertainty.
            nu (float): Scaling factor for exploration.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_arms = num_arms
        self.arm_costs = arm_costs
        self.cost_weighting = cost_weighting
        self.lambd = lambd
        self.nu = nu
        self.lr = 0.001

        self.model = CNN(num_arms).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = lambd * torch.ones(total_params).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """
        Reset cache for training data.
        """
        self.context_list = []
        self.arm_list = []
        self.reward_list = []

    def select(self, context: Union[np.ndarray, torch.Tensor], t: int
               ) -> Tuple[int, List[float], List[float]]:
        """
        Select an arm based on Thompson Sampling.

        Args:
            context (np.ndarray | torch.Tensor): Context vector.
            t (int): Time step.

        Returns:
            Tuple[int, List[float], List[float]]:
                - Selected arm index.
                - Sampled rewards (exploration adjusted).
                - True predicted rewards.
        """
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        model_outputs = self.model(context_tensor).squeeze(0)

        g_list = []
        sampled_rewards = []
        true_rewards = []

        for arm_index in range(self.num_arms):
            reward_prediction = model_outputs[arm_index]
            self.model.zero_grad()
            reward_prediction.backward(retain_graph=True)

            gradients = torch.cat([p.grad.flatten().detach() for p in self.model.parameters()])
            sigma = torch.sqrt(self.lambd * torch.sum(gradients * gradients / self.U))

            sampled_r = reward_prediction.item() + Normal(0, self.nu * sigma.item()).sample().item()
            sampled_r -= self.arm_costs[arm_index] * self.cost_weighting

            g_list.append(gradients)
            sampled_rewards.append(sampled_r)
            true_rewards.append(reward_prediction.item())

        selected_arm = int(np.argmax(sampled_rewards))
        self.U += g_list[selected_arm] * g_list[selected_arm]

        return selected_arm, sampled_rewards, true_rewards

    def update(self, context: np.ndarray, arm_idx: int, final_r: float, t: int) -> None:
        """
        Update the bandit with observed rewards.

        Args:
            context (np.ndarray): Context vector.
            arm_idx (int): Selected arm index.
            final_r (float): Final reward received.
            t (int): Time step.
        """
        self.context_list.append(context)
        self.arm_list.append(arm_idx)
        self.reward_list.append(final_r)

    def train(self, num_epochs: int = 5, batch_size: int = 8) -> float:
        """
        Train the model on stored data.

        Args:
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size.

        Returns:
            float: Average loss after training.
        """
        if not self.reward_list:
            return 0.0

        index = np.arange(len(self.reward_list))
        np.random.shuffle(index)

        total_loss = 0
        num_batches = max(1, len(self.reward_list) // batch_size)

        for epoch in range(num_epochs):
            np.random.shuffle(index)
            for batch_idx in range(num_batches):
                self.optimizer.zero_grad()
                batch_loss = 0

                for i in range(batch_size):
                    idx = index[min(batch_idx * batch_size + i, len(self.reward_list) - 1)]
                    context = torch.tensor(np.array(self.context_list[idx]), dtype=torch.float32, device=self.device).unsqueeze(0)
                    model_outputs = self.model(context).squeeze(0)

                    r = self.reward_list[idx]
                    arm = self.arm_list[idx]

                    predicted_reward = model_outputs[arm] - self.arm_costs[arm] * self.cost_weighting
                    loss = (predicted_reward - r) ** 2
                    batch_loss += loss

                batch_loss /= batch_size
                batch_loss.backward()
                self.optimizer.step()

                total_loss += batch_loss.item()

        self._initialize_cache()
        return total_loss / (num_batches * num_epochs)


# ---------------------------------------------------------------------------------------- #
#                                     REINFORCE algorithm                                  #
# ---------------------------------------------------------------------------------------- #

class Reinforce:
    """
    REINFORCE algorithm for contextual multi-arm bandits with Thompson Sampling.
    """

    def __init__(self, num_arms: int, arm_costs: List[float], cost_weighting: float,
                 lambd: float, nu: float, lr: float = 0.001, device: Optional[torch.device] = None):
        """
        Initialize the Reinforce agent.

        Args:
            num_arms (int): Number of arms.
            arm_costs (List[float]): List of arm costs.
            cost_weighting (float): Cost influence on reward.
            lambd (float): Regularization param.
            nu (float): Exploration param.
            lr (float): Learning rate.
            device (torch.device, optional): Device to run computations.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_arms = num_arms
        self.arm_costs = torch.tensor(arm_costs, device=self.device)
        self.cost_weighting = cost_weighting
        self.lambd = lambd
        self.nu = nu
        self.lr = lr

        self.model = CNN(num_arms).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self._initialize_uncertainties()
        self._initialize_cache()

    def _initialize_uncertainties(self) -> None:
        """Initialize uncertainty U for Thompson Sampling."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = self.lambd * torch.ones(total_params, device=self.device)

    def _initialize_cache(self) -> None:
        """Initialize caches for storing observations."""
        self.context_list = []
        self.arm_list = []
        self.reward_list = []

    def select(self, context: np.ndarray, t: int
               ) -> Tuple[int, List[float], List[float]]:
        """
        Select an arm using Thompson Sampling.

        Args:
            context (np.ndarray): Context input.
            t (int): Current timestep.

        Returns:
            tuple: (selected_arm, sampled_rewards, true_rewards)
        """
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        model_outputs = self.model(context_tensor).squeeze(0)

        g_list = []
        sampled_rewards = torch.zeros(self.num_arms, device=self.device)
        true_rewards = model_outputs.detach().clone()

        for arm_index in range(self.num_arms):
            self.model.zero_grad()
            reward_prediction = model_outputs[arm_index]
            reward_prediction.backward(retain_graph=True)

            gradients = torch.cat([p.grad.flatten() for p in self.model.parameters()])
            sigma = torch.sqrt(torch.clamp(self.lambd * torch.sum(gradients * gradients / self.U), min=1e-6))

            sampled_r = reward_prediction.item() + Normal(0, self.nu * sigma.item()).sample().item()
            sampled_r -= self.arm_costs[arm_index] * self.cost_weighting
            sampled_rewards[arm_index] = sampled_r

            g_list.append(gradients)

        self.g_list = g_list
        selected_arm = int(torch.argmax(sampled_rewards))

        return selected_arm, sampled_rewards.tolist(), true_rewards.tolist()

    def update(self, context: np.ndarray, arm_idx: int, final_r: float, t: int) -> None:
        """
        Update the REINFORCE policy with observed data.

        Args:
            context (np.ndarray): Context input.
            arm_idx (int): Selected arm index.
            final_r (float): Reward received.
            t (int): Current timestep.
        """
        self.context_list.append(context)
        self.arm_list.append(arm_idx)
        self.reward_list.append(final_r)

        self.U += self.g_list[arm_idx] ** 2
        self.g_list = None

    def train(self, num_epochs: int, batch_size: int) -> None:
        """
        Train REINFORCE policy.

        Args:
            num_epochs (int): Number of epochs.
            batch_size (int): Batch size.
        """
        if not self.reward_list:
            return

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = len(self.reward_list) // batch_size + 1

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(self.reward_list))

                if start_idx >= end_idx:
                    continue

                batch_contexts = torch.tensor(np.array(self.context_list[start_idx:end_idx]), dtype=torch.float32, device=self.device)
                batch_arm_indices = torch.tensor(np.array(self.arm_list[start_idx:end_idx]), dtype=torch.long, device=self.device)
                batch_rewards = torch.tensor(np.array(self.reward_list[start_idx:end_idx]), dtype=torch.float32, device=self.device)

                logits = self.model(batch_contexts)
                probs = torch.softmax(logits, dim=1)

                log_probs = torch.log(probs[range(len(batch_arm_indices)), batch_arm_indices] + 1e-6)
                loss = -torch.mean(batch_rewards * log_probs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

        self._initialize_cache()

# ---------------------------------------------------------------------------------------- #
#                               Structured REINFORCE algorithm                             #
# ---------------------------------------------------------------------------------------- #

class StructuredReinforce:
    """
    Structured REINFORCE algorithm for modular routing decisions.

    This class handles multiple independent REINFORCE policies, one for each routing decision/module.
    Each policy learns its own optimal arm independently, which enables a structured exploration 
    across different routing options.

    Attributes:
        num_modules (int): Number of routing modules.
        policies (List[Reinforce]): List of REINFORCE policies for each module.
    """

    def __init__(self, module_info: Dict[str, List[float]], cost_weighting: float, 
                 lambd: float, nu: float, lr: float = 0.001):
        """
        Initialize the StructuredReinforce router.

        Args:
            module_info (dict): Mapping of routing modules to their arm costs.
            cost_weighting (float): Weighting factor for cost in reward calculation.
            lambd (float): Regularization parameter.
            nu (float): Exploration factor.
            lr (float): Learning rate.
        """
        self.num_modules = len(module_info.keys())
        
        # Instantiate individual REINFORCE policies for each module
        self.policies = [
            Reinforce(
                num_arms=len(arm_costs),
                arm_costs=arm_costs,
                cost_weighting=cost_weighting,
                lambd=lambd,
                nu=nu,
                lr=lr
            )
            for i, arm_costs in enumerate(module_info.values())
        ]

    def select(self, context: np.ndarray, t: int
               ) -> Tuple[List[int], List[List[float]], List[List[float]]]:
        """
        Select arms for each module independently.

        Args:
            context (np.ndarray): Input context.
            t (int): Current timestep.

        Returns:
            tuple:
                - selected_arms (List[int]): Chosen arm index for each module.
                - sampled_rewards_list (List[List[float]]): Sampled rewards for each module.
                - true_rewards_list (List[List[float]]): True predicted rewards for each module.
        """
        selected_arms = []
        sampled_rewards_list = []
        true_rewards_list = []

        for policy in self.policies:
            selected_arm, sampled_rewards, true_rewards = policy.select(context, t)
            selected_arms.append(selected_arm)
            sampled_rewards_list.append(sampled_rewards)
            true_rewards_list.append(true_rewards)

        return selected_arms, sampled_rewards_list, true_rewards_list

    def update(self, context: np.ndarray, arm_indices: List[int], final_r: float, t: int) -> None:
        """
        Update each REINFORCE policy with the observed reward.

        Args:
            context (np.ndarray): Input context.
            arm_indices (List[int]): Selected arm index for each module.
            final_r (float): Reward received from the environment.
            t (int): Current timestep.
        """
        for i, policy in enumerate(self.policies):
            policy.update(context, arm_indices[i], final_r, t)

    def train(self, num_epochs: int, batch_size: int) -> None:
        """
        Train each REINFORCE policy on its stored experiences.

        Args:
            num_epochs (int): Number of epochs for training.
            batch_size (int): Size of each training batch.
        """
        for policy in self.policies:
            policy.train(num_epochs, batch_size)

# ---------------------------------------------------------------------------------------- #
#                        Parallelized Structured REINFORCE algorithm                       #
# ---------------------------------------------------------------------------------------- #

class ParallizedStructuredReinforce:
    """
    Parallelized Structured REINFORCE algorithm for distributed routing decisions.

    Each REINFORCE policy is assigned to a separate GPU (if available) to parallelize
    training and improve performance. Arm selection and updates are performed sequentially,
    while training is parallelized across GPUs.

    Attributes:
        num_modules (int): Number of routing modules.
        devices (List[torch.device]): Devices assigned to each policy.
        policies (List[Reinforce]): List of REINFORCE policies for each module.
    """

    def __init__(self, module_info: Dict[str, List[float]], cost_weighting: float, 
                 lambd: float, nu: float, lr: float = 0.001):
        """
        Initialize the Parallelized StructuredReinforce router.

        Args:
            module_info (dict): Mapping of routing modules to their arm costs.
            cost_weighting (float): Weighting factor for cost in reward calculation.
            lambd (float): Regularization parameter.
            nu (float): Exploration factor.
            lr (float): Learning rate.
        """
        self.num_modules = len(module_info.keys())
        
        # Distribute policies across available GPUs
        num_gpus = torch.cuda.device_count()
        self.devices = [
            torch.device(f'cuda:{i % num_gpus}') if num_gpus > 0 else torch.device('cpu')
            for i in range(self.num_modules)
        ]

        # Instantiate individual REINFORCE policies for each module on its device
        self.policies = [
            Reinforce(
                num_arms=len(arm_costs),
                arm_costs=arm_costs,
                cost_weighting=cost_weighting,
                lambd=lambd,
                nu=nu,
                lr=lr,
                device=self.devices[i]
            )
            for i, arm_costs in enumerate(module_info.values())
        ]

    def select(self, context: np.ndarray, t: int
               ) -> Tuple[List[int], List[List[float]], List[List[float]]]:
        """
        Select arms for each module independently.

        Args:
            context (np.ndarray): Input context.
            t (int): Current timestep.

        Returns:
            tuple:
                - selected_arms (List[int]): Chosen arm index for each module.
                - sampled_rewards_list (List[List[float]]): Sampled rewards for each module.
                - true_rewards_list (List[List[float]]): True predicted rewards for each module.
        """
        selected_arms = []
        sampled_rewards_list = []
        true_rewards_list = []

        for policy in self.policies:
            selected_arm, sampled_rewards, true_rewards = policy.select(context, t)
            selected_arms.append(selected_arm)
            sampled_rewards_list.append(sampled_rewards)
            true_rewards_list.append(true_rewards)

        return selected_arms, sampled_rewards_list, true_rewards_list

    def update(self, context: np.ndarray, arm_indices: List[int], final_r: float, 
               reward_mapping: List[int], t: int) -> None:
        """
        Update each REINFORCE policy with the observed reward, based on reward mapping.

        Args:
            context (np.ndarray): Input context.
            arm_indices (List[int]): Selected arm index for each module.
            final_r (float): Reward received from the environment.
            reward_mapping (List[int]): Mapping of rewards for each module, 1 for reward, 0 for no reward.
            t (int): Current timestep.
        """
        for i, policy in enumerate(self.policies):
            if reward_mapping[i] == 1:
                policy.update(context, arm_indices[i], final_r, t)

    def train(self, num_epochs: int, batch_size: int) -> None:
        """
        Train each REINFORCE policy in parallel.

        Args:
            num_epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        """
        # Parallelize training across multiple GPUs using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(policy.train, num_epochs, batch_size)
                for policy in self.policies
            ]

            for future in as_completed(futures):
                try:
                    future.result()  # Raise any exceptions caught during training
                except Exception as e:
                    print(f"Training failed for a policy: {e}")
