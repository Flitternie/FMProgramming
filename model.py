import numpy as np
from data import measurer

module_costs = {
    'functionA': [9, 18, 36],  # GLIPv2-T, GLIPv2-B, GLIPv2-H
    'functionB': [40, 120],    # BLIP2-t5-xl, BLIP2-t5-xxl
}

def model_performance(input, mean, std):
    # Get the complexity score from the complexity API
    input_complexity = measurer.complexity(input)
    # Generate performance score by adding Gaussian noise to the complexity score
    # Noise is centered around (mean - input_complexity) to shift the mean performance to mean
    performance = input_complexity + np.random.normal(mean - input_complexity, std)
    # Ensure the performance score is within the bounds of 0 and 1
    performance_clamped = max(0, min(1, performance))
    return performance_clamped

def model_a1(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # cheap, low-quality implementation
    # return bool(np.random.choice([True, False], p=[0.6, 0.4]))
    performance = model_performance(input[0], mean=0.5, std=0.3)
    if performance < 0.5:
        return False
    else:
        return True

def model_a2(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # medium, medium-quality implementation
    # return bool(np.random.choice([True, False], p=[0.7, 0.3]))
    performance = model_performance(input[0], mean=0.7, std=0.3)
    if performance < 0.5:
        return False
    else:
        return True

def model_a3(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # expensive, high-quality implementation
    # return bool(np.random.choice([True, False], p=[0.8, 0.2]))
    performance = model_performance(input[0], mean=0.9, std=0.3)
    if performance < 0.5:
        return False
    else:
        return True

def model_b1(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # cheap, low-quality implementation
    # return bool(np.random.choice([True, False], p=[0.7, 0.3]))
    performance = model_performance(input[0], mean=0.7, std=0.3)
    if performance < 0.5:
        return False
    else:
        return True


def model_b2(input):
    if isinstance(input[-1], bool) and not input[-1]:
        return False
    # expensive, high-quality implementation
    # return bool(np.random.choice([True, False], p=[0.8, 0.2]))
    performance = model_performance(input[0], mean=0.9, std=0.3)
    if performance < 0.5:
        return False
    else:
        return True
    