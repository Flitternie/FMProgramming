import os
import numpy as np
from tqdm import tqdm

from data import measurer, get_dataset
from bandit import MuFasa, UCB1, UCB_ALP, Contextual
from execute import idx_to_arms, routing
from utils import *

modules = {
    'functionA': 3,
    'functionB': 2,
}


def pseudo_loss(pred, target):
    return (pred - target) ** 2

def get_reward(exec_result, exec_cost, weight=0.00001):
    return - pseudo_loss(int(exec_result), 1) - weight * exec_cost

def baseline():
    algo = UCB1(modules)
    for arm_idx in range(algo.num_arms):
        set_seed(42)
        program = [("functionA", None), ("functionB", None)]
        dataset = get_dataset()
        selected_arms = idx_to_arms(arm_idx, modules)
        correct = 0
        total_cost = 0
        total_len = 1000
        pbar = tqdm(total=total_len)
        for t in range(total_len):
            context = dataset.step()
            exec_result, exec_cost = routing(context, program, selected_arms)
            correct += int(exec_result)
            total_cost += exec_cost
            pbar.set_description(f"Arm: {selected_arms}, Accuracy: {correct/(t+1)}, Average cost: {total_cost/(t+1)}")
            pbar.update(1)
        print(f"Arm: {selected_arms}, Accuracy: {correct/total_len}, Average cost: {total_cost/total_len}")
        with open("./logs/baseline.txt", "a") as f:
            f.write(f"{selected_arms}; {correct/total_len}; {total_cost/total_len}\n")


def ucb1(cost_weighting_list, save_dir="./logs/", arg_nu=1):
    # hyperparameters

    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        
        set_seed(42)
        dataset = get_dataset()
        
        program = [("functionA", None), ("functionB", None)]
        algo = UCB1(modules, arg_nu)

        gathered_reward = []
        total_correct = 0
        total_cost = 0
        log = []

        total_len = 1000
        pbar = tqdm(total=total_len)
        for t in range(total_len):
            context = dataset.step()
            arm_idx = algo.select(context, t) # for UCB1 and UCB_ALP
            if arm_idx is None:
                raise ValueError("Invalid arm index")
            selected_arms = idx_to_arms(arm_idx, modules)
            # Execute the program with the selected arms
            exec_result, exec_cost = routing(context, program, selected_arms)
            total_correct += int(exec_result)
            total_cost += exec_cost
            final_reward = get_reward(exec_result, exec_cost, arg_cost_weighting)
            gathered_reward.append(final_reward)
            # algo.update(context, final_reward) # for MuFasa
            algo.update(arm_idx, final_reward) # for UCB1 and UCB_ALP    
            
            # if t % 5 == 0:
            #     loss = algo.train()
            loss = 0 # for UCB1 and UCB_ALP
            
            avg_accuracy, avg_cost = total_correct/(t+1), total_cost/(t+1)
            log.append([t, 0, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward)])
            pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}, {:.5f}".format(avg_accuracy, avg_cost, loss))
            pbar.update(1)

        pbar.close()
        print("Accuracy: ", total_correct/len(dataset))
        print("Average cost: ", total_cost/len(dataset))
        with open(os.path.join(save_dir, np.format_float_positional(arg_cost_weighting) + ".txt"), "w") as f:
            for l in log:
                f.write(", ".join([str(i) for i in l]) + "\n")

        print(f"**** Finished processing cost weighting: {arg_cost_weighting} ****")

def contextual(cost_weighting_list, save_dir="./logs/"):
    # hyperparameters
    arg_lambda = 0.0001 # Regularization parameter
    arg_nu = 2 # Exploration-exploitation trade-off parameter
    arg_hidden_size = 128 # Hidden size of the neural network

    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        
        set_seed(42)
        dataset = get_dataset()
        input_size = 1
        
        program = [("functionA", None), ("functionB", None)]
        algo = Contextual(modules, input_size, arg_lambda, arg_nu, arg_hidden_size)

        gathered_reward = []
        total_correct = 0
        total_cost = 0
        log = []

        total_len = 1000
        pbar = tqdm(total=total_len)
        for t in range(total_len):
            context = dataset.step()
            context_complexiy = measurer.complexity(context)
            input_feature = np.array([context_complexiy])
            arm_idx = algo.select(input_feature, t) 
            if arm_idx is None:
                raise ValueError("Invalid arm index")
            selected_arms = idx_to_arms(arm_idx, modules)
            selected_arms = [0,0]
            # Execute the program with the selected arms
            exec_result, exec_cost = routing(context, program, selected_arms)
            total_correct += int(exec_result)
            total_cost += exec_cost
            final_reward = get_reward(exec_result, exec_cost, arg_cost_weighting)
            gathered_reward.append(final_reward)
            algo.update(input_feature, arm_idx, final_reward, t) # for MuFasa
            
            if t % 5 == 0:
                loss = algo.train()
            
            avg_accuracy, avg_cost = total_correct/(t+1), total_cost/(t+1)
            log.append([t, loss, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward)])
            pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}, {:.5f}".format(avg_accuracy, avg_cost, loss))
            pbar.update(1)

        pbar.close()
        print("Accuracy: ", total_correct/len(dataset))
        print("Average cost: ", total_cost/len(dataset))
        with open(os.path.join(save_dir, np.format_float_positional(arg_cost_weighting) + ".txt"), "w") as f:
            for l in log:
                f.write(", ".join([str(i) for i in l]) + "\n")

        print(f"**** Finished processing cost weighting: {arg_cost_weighting} ****")

if __name__ == "__main__":
    cost_weighting_list = [0.001]
    # cost_weighting_list += [-0.5, -0.1, -0.05, -0.01, -0.005, -0.001]
    # cost_weighting_list += [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    # cost_weighting_list = [0.02, 0.03, 0.04] + [0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009]
    # cost_weighting_list = [0.001 + 0.0002 * i for i in range(1, 10)]
    print(cost_weighting_list)
    
    save_dir = "./logs/contextual/"
    # Create the logs directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    contextual(cost_weighting_list, save_dir)



    # save_dir = "./logs/ucb1_new/"
    # # Create the logs directory if it does not exist
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    # ucb1(cost_weighting_list, save_dir, 0.5)
    
    # baseline()