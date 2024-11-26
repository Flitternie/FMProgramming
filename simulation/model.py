import numpy as np
from simulation.data import measurer, get_dataset, num_sample
import random

'''
This file contains the code to define the pool of available models for each functions and their corresponding performance approximations.
'''

module_costs = {
    'functionA': [9, 27, 36],  # GLIPv2-T, GLIPv2-B, GLIPv2-H
    'functionB': [40, 120],    # BLIP2-t5-xl, BLIP2-t5-xxl
    'functionC': [36.5, 42.7, 93.8, 100],
}

def decay_function(x, type='logistic', k=10, x0=1, ymin=0.3, ymax=1, noise=0.05):
    '''
    This function returns a value between ymin and ymax based on the input x, 
    using a logistic or exponential decay function, 
    with random noise sampled from a normal distribution.

    Parameters:
    x (float): input value
    type (str): type of decay function ('logistic' or 'exponential')
    k (float): steepness of the curve
    x0 (float): inflection point, where the curve changes direction, for logistic function
    ymin (float): minimum value
    ymax (float): maximum value
    noise (float): standard deviation of the noise
    '''
    x = 1 - x
    if type == 'logistic':
        return np.clip(ymax - (ymax - ymin) / (1 + np.exp(k * (x - x0))) + np.random.normal(0, noise**(x), 1), ymin, 1)
    elif type == 'exponential':
        return np.clip(ymax - (ymax - ymin) * np.exp(-k * x) + np.random.normal(0, noise**(x), 1), ymin, 1)
    else:
        raise ValueError("Unsupported type. Use 'logistic' or 'exponential'.")
    

def model_a1(input):
    complexity = measurer.complexity(input[0])
    # Accuracy:  0.7022161832291042
    result = decay_function(complexity, type='logistic', k=10, x0=0.5, ymin=0.1, ymax=1)[0]
    if isinstance(input[-1], float):
        return result * input[-1]
    return result

def model_a2(input):
    complexity = measurer.complexity(input[0])
    # Accuracy:  0.7617358942191104
    result = decay_function(complexity, type='logistic', k=10, x0=0.48, ymin=0.1, ymax=1)[0]
    if isinstance(input[-1], float):
        return result * input[-1]
    return result

def model_a3(input):
    complexity = measurer.complexity(input[0])
    # Accuracy:  0.8555830823938965
    result = decay_function(complexity, type='logistic', k=10, x0=0.35, ymin=0.1, ymax=1)[0]
    if isinstance(input[-1], float):
        return result * input[-1]
    return result



def model_b1(input):
    complexity = measurer.complexity(input[0])
    # Accuracy:  0.7092067980551909
    result = decay_function(complexity, type='exponential', k=2, ymin=0.1, ymax=1)[0]
    if isinstance(input[-1], float):
        return result * input[-1]
    return result

def model_b2(input):
    complexity = measurer.complexity(input[0])
    # Accuracy:  0.9132142918962406
    result = decay_function(complexity, type='exponential', k=4, ymin=0.3, ymax=1)[0]
    if isinstance(input[-1], float):
        return result * input[-1]
    return result



def model_c1(input):
    complexity = measurer.complexity(input[0])
    result = decay_function(complexity, type='logistic', k=7, x0=0.5, ymin=0.1, ymax=0.7, noise=0.01)[0]
    result = 1 if result > 0.5 else 0
    if isinstance(input[-1], float):
        return result * input[-1]
    return result

def model_c2(input):
    complexity = measurer.complexity(input[0])
    result = decay_function(complexity, type='logistic', k=10, x0=0.4, ymin=0.3, ymax=1, noise=0.01)[0]
    result = 1 if result > 0.5 else 0
    if isinstance(input[-1], float):
        return result * input[-1]
    return result

def model_c3(input):
    complexity = measurer.complexity(input[0])
    result = decay_function(complexity, type='exponential', k=5, ymin=0.1, ymax=0.8, noise=0.01)[0]
    result = 1 if result > 0.5 else 0   
    if isinstance(input[-1], float):
        return result * input[-1]
    return result

def model_c4(input):
    complexity = measurer.complexity(input[0])
    result = decay_function(complexity, type='exponential', k=10, ymin=0.3, ymax=1, noise=0.01)[0]
    result = 1 if result > 0.5 else 0
    if isinstance(input[-1], float):
        return result * input[-1]
    return result