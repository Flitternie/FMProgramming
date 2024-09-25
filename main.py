import os
import sys
import numpy as np
from tqdm import tqdm

from data import measurer, get_dataset, num_sample
from bandit import MuFasa, UCB1, UCB_ALP, Contextual
from execute import idx_to_arms, routing
from neural_bandit.neuralts import NeuralTS
from neural_bandit.neuralucb import NeuralUCB

from utils import *

total_len = num_sample

modules = {
    'functionA': 3,
    'functionB': 2,
    'functionC': 4,
}

# Specify the program to be executed
# program = [("functionA", None), ("functionB", None)]
program = [("functionC", None)]

exp_name = ""

# Calculate the total number of arms given the program
num_arms = 1
for i in program:
    num_arms *= modules[i[0]]


def baseline():
    '''
    This function runs the baseline experiment, where the program is executed with all possible combinations of arms.
    '''
    for arm_idx in range(num_arms):
        set_seed(7)
        dataset = get_dataset()
        selected_arms = idx_to_arms(arm_idx, modules)
        correct = 0
        total_cost = 0
        pbar = tqdm(total=total_len)
        with open(f"./logs/{exp_name}{arm_idx}_results.txt", "w") as f:
            for t in range(total_len):
                context = dataset.step()
                complexity = measurer.complexity(context)
                exec_result, exec_cost = routing(context, program, selected_arms)
                correct += exec_result
                total_cost += exec_cost
                pbar.set_description(f"Arm: {selected_arms}, Accuracy: {correct/(t+1)}, Average cost: {total_cost/(t+1)}")
                pbar.update(1)
                f.write(f"{t}; {complexity}; {exec_result}; {exec_cost}; \n")
        print(f"Arm: {selected_arms}, Accuracy: {correct/total_len}, Average cost: {total_cost/total_len}")
        with open("./logs/baseline.txt", "a") as f:
            f.write(f"{selected_arms}; {correct/total_len}; {total_cost/total_len}\n")

# Load the baseline results, if failed, run the baseline
try:
    baseline_results = {}
    baseline_costs = {}
    for i in range(num_arms):
        with open(f"./logs/{exp_name}{i}_results.txt", "r") as f:
            lines = f.readlines()
            baseline_results[i] = []
            for line in lines:
                line = line.strip()
                baseline_results[i].append(float(line.split("; ")[2]))
            baseline_costs[i] = float(lines[-1].split("; ")[-2])
except:
    print("Baseline results not found, running the baseline")
    baseline()


def pseudo_loss(pred, target):
    '''
    This function calculates the pseudo loss, which is the square of the difference between the prediction and the target.
    '''
    return (pred - target) ** 2


def get_reward(context, program, arm_idx, weight, t):
    '''
    This function calculates the reward for a given arm index, based on the context, program, and cost weighting.
    '''
    # selected_arms = idx_to_arms(arm_idx, modules)
    # exec_result, exec_cost = routing(context, program, selected_arms)
    exec_result = baseline_results[arm_idx][t]
    exec_cost = baseline_costs[arm_idx]
    return exec_result, exec_cost, - (pseudo_loss(exec_result, 1) + weight * exec_cost) * 100
        

def oracle(cost_weighting_list, save_dir="./logs/"):
    '''
    This function runs the oracle experiment, where the optimal arm is selected based on the true reward.
    '''
    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        set_seed(7)
        dataset = get_dataset()
        gathered_reward = []
        total_correct = 0
        total_cost = 0
        pbar = tqdm(total=total_len)
        f = open(os.path.join(save_dir, np.format_float_positional(arg_cost_weighting) + ".txt"), "w", buffering=1)
        for t in range(total_len):
            context = dataset.step()
            rewards = []
            for idx in range(num_arms):
                _, _, reward = get_reward(context, program, idx, arg_cost_weighting, t)
                rewards.append(reward)
            
            arm_idx = np.argmax(rewards)
            exec_result, exec_cost, final_reward = get_reward(context, program, arm_idx, arg_cost_weighting, t)
            total_correct += exec_result
            total_cost += exec_cost
            gathered_reward.append(final_reward)            
            avg_accuracy, avg_cost = total_correct/(t+1), total_cost/(t+1)
            
            line = "; ".join(str(i) for i in [t, 0, measurer.complexity(context), arm_idx, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward)]) + "\n"
            f.write(line)
            pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}".format(avg_accuracy, avg_cost))
            pbar.update(1)
        
        f.close()
        pbar.close()
        print("Accuracy: ", total_correct/len(dataset))
        print("Average cost: ", total_cost/len(dataset))
        
        print(f"**** Finished processing cost weighting: {arg_cost_weighting} ****")


def ucb1(cost_weighting_list, save_dir="./logs/", arg_nu=1):
    '''
    This function runs the UCB1 experiment, where the arm is selected based on the UCB1 algorithm.
    '''
    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        
        set_seed(42)
        dataset = get_dataset()
        
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
    '''
    This function runs the contextual bandit experiment, where the arm is selected based on the Contextual Bandit algorithm.
    '''
    # hyperparameters
    arg_lambda = 0.0001 # Regularization parameter
    arg_nu = 0.1 # Exploration-exploitation trade-off parameter
    arg_hidden_size = 128 # Hidden size of the neural network

    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        
        set_seed(7)
        dataset = get_dataset()
        input_size = 384
        
        # algo = NeuralUCB(input_size, num_arms, beta=1, lamb=1)
        algo = Contextual(modules, input_size, arg_lambda, arg_nu, arg_hidden_size)
        # algo = MuFasa(modules, input_size, arg_lambda, arg_nu, arg_hidden_size)

        gathered_reward = []
        regrets = []
        total_correct = 0
        total_cost = 0

        pbar = tqdm(total=total_len)
        f = open(os.path.join(save_dir, np.format_float_positional(arg_cost_weighting) + ".txt"), "w", buffering=1)
        for t in range(total_len):
            context = dataset.step()
            # context_complexiy = measurer.complexity(context)
            input_feature = np.array(context)
            
            # arm_idx = algo.take_action(input_feature)
            arm_idx, _, _ = algo.select(input_feature, t) 

            if arm_idx is None:
                raise ValueError("Invalid arm index")
            
            selected_arms = idx_to_arms(arm_idx, modules)
            # Execute the program with the selected arms
            exec_result, exec_cost, final_reward = get_reward(context, program, arm_idx, arg_cost_weighting, t)
            
            rewards = []
            for idx in range(num_arms):
                _, _, reward = get_reward(context, program, idx, arg_cost_weighting, t)
                rewards.append(reward)
            regret = max(rewards) - final_reward
            regrets.append(regret)

            total_correct += exec_result
            total_cost += exec_cost
            gathered_reward.append(final_reward)

            # algo.update(context, arm_idx, final_reward)
            algo.update(input_feature, arm_idx, final_reward, t)

            
            if t % 8 == 0:
                loss = algo.train()
            
            avg_accuracy, avg_cost = total_correct/(t+1), total_cost/(t+1)
            avg_regret = sum(regrets)/len(regrets)
            try:
                line = "; ".join(str(i) for i in [t, loss, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward), avg_regret]) + "\n"
                pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}, Loss: {:3f}, Regret {:.3f}".format(avg_accuracy, avg_cost, loss, avg_regret))
            except:
                line = "; ".join(str(i) for i in [t, 0, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward), avg_regret]) + "\n"
                pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}, Average regret: {:.3f}".format(avg_accuracy, avg_cost, avg_regret))
            f.write(line)
            pbar.update(1)
        
        f.close()
        pbar.close()
        print("Accuracy: ", total_correct/len(dataset))
        print("Average cost: ", total_cost/len(dataset))
        
        print(f"**** Finished processing cost weighting: {arg_cost_weighting} ****")


def neural(cost_weighting_list, save_dir="./logs/"):
    '''
    This function runs the neural bandit experiment, where the arm is selected based on the neural_bandit implementation at https://github.com/wadx2019/Neural-Bandit
    '''
    # hyperparameters
    arg_lambda = 0.0001 # Regularization parameter
    arg_nu = 0.1 # Exploration-exploitation trade-off parameter
    arg_hidden_size = 128 # Hidden size of the neural network

    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        
        set_seed(7)
        dataset = get_dataset()
        input_size = 384
        
        # algo = NeuralUCB(input_size, num_arms, beta=1, lamb=1)
        algo = NeuralTS(input_size, num_arms, beta=1, lamb=1)

        gathered_reward = []
        regrets = []
        total_correct = 0
        total_cost = 0

        pbar = tqdm(total=total_len)
        f = open(os.path.join(save_dir, np.format_float_positional(arg_cost_weighting) + ".txt"), "w", buffering=1)
        for t in range(total_len):
            context = dataset.step()
            input_feature = np.array(context)
            
            arm_idx = algo.take_action(input_feature)

            if arm_idx is None:
                raise ValueError("Invalid arm index")
            
            selected_arms = idx_to_arms(arm_idx, modules)
            # Execute the program with the selected arms
            exec_result, exec_cost, final_reward = get_reward(context, program, arm_idx, arg_cost_weighting, t)
            
            rewards = []
            for idx in range(num_arms):
                _, _, reward = get_reward(context, program, idx, arg_cost_weighting, t)
                rewards.append(reward)
            regret = max(rewards) - final_reward
            regrets.append(regret)

            total_correct += exec_result
            total_cost += exec_cost
            gathered_reward.append(final_reward)

            algo.update(context, arm_idx, final_reward)

            
            if t % 8 == 0:
                loss = algo.train()
            
            avg_accuracy, avg_cost = total_correct/(t+1), total_cost/(t+1)
            avg_regret = sum(regrets)/len(regrets)
            try:
                line = "; ".join(str(i) for i in [t, loss, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward), avg_regret]) + "\n"
                pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}, Loss: {:3f}, Regret {:.3f}".format(avg_accuracy, avg_cost, loss, avg_regret))
            except:
                line = "; ".join(str(i) for i in [t, 0, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward), avg_regret]) + "\n"
                pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}, Average regret: {:.3f}".format(avg_accuracy, avg_cost, avg_regret))
            f.write(line)
            pbar.update(1)
        
        f.close()
        pbar.close()
        print("Accuracy: ", total_correct/len(dataset))
        print("Average cost: ", total_cost/len(dataset))
        
        print(f"**** Finished processing cost weighting: {arg_cost_weighting} ****")


if __name__ == "__main__":
    # Detect argv for experiment name
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        print("Experiment name: ", exp_name)

    # List of cost weightings to test
    cost_weighting_list = [0.0]
    cost_weighting_list += [0.0005, 0.001, 0.005, 0.01, 0.05]
    cost_weighting_list += [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    
    # Test the oracle
    # save_dir = f"./logs/{exp_name}oracle/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    # oracle(cost_weighting_list, save_dir)

    # Test the UCB1
    # save_dir = "./logs/{exp_name}ucb1/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    # ucb1(cost_weighting_list, save_dir, 0.5)

    # Test the contextual bandit
    # save_dir = f"./logs/{exp_name}contextual/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    # contextual(cost_weighting_list, save_dir)

    # Test the neural bandit
    save_dir = f"./logs/{exp_name}neural/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    neural(cost_weighting_list, save_dir)

    # Test the baseline
    # baseline()