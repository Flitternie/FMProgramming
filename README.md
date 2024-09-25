# A Routing System for Agent Program Cost-efficiency Optimization

## Introduction
This project is a cost-efficient routing system designed to automatically optimize an agent program by learning to select the cost-effcient implementation for each function. 

## How to run
Read the header of each files for their purpose and run the main.py file to see the results.
- `data.py` contains the data generation and data processing functions. Download `imdb_preprocessed.csv` from [here](https://drive.google.com/file/d/1vQR7pqVqmYndHkfpcEpwCSWYDXON0gkl/view?usp=sharing) and put it in the working directory.
- `model.py` contains the simulation of different models' performance based on the input complexity and the cost of the model.
- `execute.py` contains the routing and execution mechanism, which routes the function to the specific model defined in `model.py` based on the decision of the bandit algorithm.
- `bandit.py` contains the bandit algorithms' implementation.
- `network.py` contains the network structure for the Multi-facet Contextual Bandit algorithm.
- `main.py` contains the main function to run the algorithm with the given data and program. 
