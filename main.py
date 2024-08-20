import os
import numpy as np
from tqdm import tqdm

from data import measurer, get_dataset
from bandit import MuFasa, UCB1, UCB_ALP, UCB_Contextual
from execute import idx_to_arms, routing
from functionality import module_costs
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
    

def main(cost_weighting_list, save_dir="./logs/"):
    # hyperparameters
    arg_lambda = 0.0001 # Regularization parameter
    arg_nu = 1 # Exploration-exploitation trade-off parameter
    arg_hidden_size = 128 # Hidden size of the neural network
    arg_cost_weighting = 0.001

    for arg_cost_weighting in cost_weighting_list:
        print("Cost weighting: ", arg_cost_weighting)
        
        set_seed(42)
        dataset = get_dataset()
        input_size = dataset.dim
        
        program = [("functionA", None), ("functionB", None)]
        # algo = MuFasa(modules, input_size, arg_lambda, arg_nu, arg_hidden_size)
        algo = UCB_Contextual(modules, input_size, arg_nu)
        # algo = UCB_ALP(modules, modules_costs, 8000, 100)
        # algo = UCB1(modules, arg_nu)

        gathered_reward = []
        total_correct = 0
        total_cost = 0
        log = []

        total_len = 1000
        pbar = tqdm(total=total_len)
        for t in range(total_len):
            context = dataset.step()
            arm_idx, nrm, sig, avg_reward = algo.select(context, t)
            # arm_idx = algo.select(t)
            if arm_idx is None:
                raise ValueError("Invalid arm index")
            selected_arms = idx_to_arms(arm_idx, modules)
            # Execute the program with the selected arms
            exec_result, exec_cost = routing(context, program, selected_arms)
            total_correct += int(exec_result)
            total_cost += exec_cost
            final_reward = get_reward(exec_result, exec_cost, arg_cost_weighting)
            gathered_reward.append(final_reward)
            # algo.update(context, final_reward, t)    
            algo.update(arm_idx, final_reward, context)
            # algo.update(arm_idx, final_reward)
            
            # if t % 5 == 0:
            #     loss = algo.train()
            # if t % 100 == 0:
            #     print('{}: {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, loss, nrm, sig, avg_reward))
            
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
    # cost_weighting_list = [0.001 * i for i in range(1, 10)] + [0.01 + 0.002 * i for i in range(1, 10)]
    cost_weighting_list = [0.0001]
    print(cost_weighting_list)
    save_dir = "./logs/mufasa/"
    # Create the logs directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    main(cost_weighting_list, save_dir)
    # baseline()