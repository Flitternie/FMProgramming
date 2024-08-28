import numpy as np
from data import measurer, get_dataset, num_sample
import random

module_costs = {
    'functionA': [9, 27, 36],  # GLIPv2-T, GLIPv2-B, GLIPv2-H
    'functionB': [40, 120],    # BLIP2-t5-xl, BLIP2-t5-xxl
    'functionC': [40, 60],  
}

# def performance_approximator(input, mean):
#     probability_of_one = mean * 1.8 * (1 - measurer.complexity(input))
    
#     # Generate a random number and compare it to the probability
#     if random.random() < probability_of_one:
#         return True
#     else:
#         return False

class ComplexityScoreTransformer:
    def __init__(self, x, m, tolerance=1e-6, max_iterations=1000):
        """
        Initialize the transformer with input complexity scores and desired mean.
        
        Parameters:
        x (array-like): Array of complexity scores, values between 0 and 1.
        m (float): The expected mean of the output values, between 0 and 1.
        tolerance (float): The tolerance for convergence to the expected mean value m.
        max_iterations (int): Maximum number of iterations for the adjustment process.
        """
        self.x = np.array(x)
        self.m = m
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.mapping = {}

        # Validate input
        if not np.all((self.x >= 0) & (self.x <= 1)):
            raise ValueError("All complexity scores must be in the range [0, 1].")
        if not (0 <= self.m <= 1):
            raise ValueError("Expected mean value m must be in the range [0, 1].")
        
        # Generate the initial negatively correlated values
        self._generate_negatively_correlated_values()

    def _generate_negatively_correlated_values(self):
        """
        Generate and store output values that are negatively correlated with input scores.
        """
        y = 1 - self.x
        current_mean = y.mean()
        iteration = 0
        
        while abs(current_mean - self.m) > self.tolerance and iteration < self.max_iterations:
            adjustment_factor = self.m / current_mean if current_mean != 0 else 1
            y = y * adjustment_factor
            y = np.clip(y, 0, 1)
            current_mean = y.mean()
            iteration += 1
        
        if iteration == self.max_iterations:
            print("Warning: Maximum iterations reached without achieving the desired mean.")
        
        # Store the mapping from x to y
        self.mapping = dict(zip(self.x, y))
    
    def get_output_value(self, x_value):
        """
        Get the output value corresponding to a given input complexity score.
        
        Parameters:
        x_value (float): A single complexity score.

        Returns:
        float: The corresponding output value, or None if x_value is not in the original input list.
        """
        if x_value in self.mapping:
            return self.mapping[x_value]
        else:
            print(f"Input value {x_value} not found in the original list of complexity scores.")
            return None

data = get_dataset()
inputs = []
for i in range(num_sample):
    x = data.step()
    complexity = measurer.complexity(x)
    inputs.append(complexity)

model_a1_estimator = ComplexityScoreTransformer(inputs, 0.7)
def model_a1(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # cheap, low-quality implementation
    return performance_approximator(input[0], 0.7)

def model_a2(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # medium, medium-quality implementation
    # return bool(np.random.choice([True, False], p=[0.7, 0.3]))
    return performance_approximator(input[0], 0.75)

def model_a3(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # expensive, high-quality implementation
    # return bool(np.random.choice([True, False], p=[0.8, 0.2]))
    return performance_approximator(input[0], 0.9)


def model_b1(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # cheap, low-quality implementation
    # return bool(np.random.choice([True, False], p=[0.7, 0.3]))
    return performance_approximator(input[0], 0.7)

def model_b2(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # expensive, high-quality implementation
    # return bool(np.random.choice([True, False], p=[0.8, 0.2]))
    return performance_approximator(input[0], 0.9)


def model_c1(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # cheap, low-quality implementation
    # return bool(np.random.choice([True, False], p=[0.7, 0.3]))
    return performance_approximator(input[0], 0.7)

def model_c2(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # expensive, high-quality implementation
    # return bool(np.random.choice([True, False], p=[0.8, 0.2]))
    return performance_approximator(input[0], 0.9) 