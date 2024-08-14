import numpy.random as random
import datasets
import pandas as pd
from data import measurer

def model_a1(input):
    if input[-1] is False:
        output = False
    # cheap, low-quality implementation
    if measurer.complexity(input[0]) > 0.5:
        # for complex inputs, 50% chance of returning False
        output = random.choice([True, False], p=[0.5, 0.5])
    else:
        # for simple inputs, 20% chance of returning False
        output = random.choice([True, False], p=[0.8, 0.2])
    return output

def model_a2(input):
    if input[-1] is False:
        output = False
    # medium, medium-quality implementation
    if measurer.complexity(input[0]) > 0.5:
        # for complex inputs, 30% chance of returning False
        output = random.choice([True, False], p=[0.7, 0.3])
    else:
        # for simple inputs, 10% chance of returning False
        output = random.choice([True, False], p=[0.9, 0.1])
    return output

def model_a3(input):
    if input[-1] is False:
        output = False
    # expensive, high-quality implementation
    if measurer.complexity(input[0]) > 0.5:
        # for complex inputs, 10% chance of returning False
        output = random.choice([True, False], p=[0.9, 0.1])
    else:
        # for simple inputs, 5% chance of returning False
        output = random.choice([True, False], p=[0.95, 0.05])
    return output

def model_b1(input):
    if input[-1] is False:
        output = False
    # cheap, low-quality implementation
    if measurer.complexity(input[0]) > 0.7:
        # for complex inputs, 30% chance of returning False
        output = random.choice([True, False], p=[0.7, 0.3])
    else:
        # for simple inputs, 20% chance of returning False
        output = random.choice([True, False], p=[0.8, 0.2])
    return output

def model_b2(input):
    if input[-1] is False:
        output = False
    # expensive, high-quality implementation
    if measurer.complexity(input[0]) > 0.7:
        # for complex inputs, 10% chance of returning False
        output = random.choice([True, False], p=[0.9, 0.1])
    else:
        # for simple inputs, returning True
        output = True
    return output