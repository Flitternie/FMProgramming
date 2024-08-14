import numpy as np 
from tqdm import tqdm

from data import measurer, get_dataset
from bandit import MuFasa, UCB1
from execute import idx_to_arms, routing
from utils import *

modules = {
    'functionA': 3,
    'functionB': 2,
    # 'functionC': 4
}

def pseudo_loss(pred, target):
    return (pred - target) ** 2

def get_reward(exec_result, exec_cost, weight=0.00001):
    return - pseudo_loss(int(exec_result), 1) - weight * exec_cost

def main():
    # hyperparameters
    arg_lambda = 0.0001 # Regularization parameter
    arg_nu = 1 # Exploration-exploitation trade-off parameter
    arg_hidden_size = 128 # Hidden size of the neural network
    arg_cost_weighting = 0.0001

    # for arg_cost_weighting in [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
    print("Cost weighting: ", arg_cost_weighting)
    
    set_seed(42)
    dataset = get_dataset()
    input_size = dataset.dim
    
    program = [("functionA", None), ("functionB", None)]
    # algo = MuFasa(modules, input_size, arg_lambda, arg_nu, arg_hidden_size)
    algo = UCB1(modules)

    gathered_reward = []
    total_correct = 0
    total_cost = 0
    log = []

    total_len = 500
    pbar = tqdm(total=total_len)
    for t in range(total_len):
        context = dataset.step()
        # arm_idx, nrm, sig, avg_reward = algo.select(context, t)
        arm_idx = algo.select()
        selected_arms = idx_to_arms(arm_idx, modules)
        # Execute the program with the selected arms
        exec_result, exec_cost = routing(context, program, selected_arms)
        total_correct += int(exec_result)
        total_cost += exec_cost
        final_reward = get_reward(exec_result, exec_cost, arg_cost_weighting)
        gathered_reward.append(final_reward)
        # algo.update(context, final_reward, t)    
        algo.update(arm_idx, final_reward)
        
        # if t % 2 == 0:
        #     loss = algo.train()
        # if t % 100 == 0:
        #     print('{}: {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, loss, nrm, sig, avg_reward))
        
        avg_accuracy, avg_cost = total_correct/(t+1), total_cost/(t+1)
        log.append([t, 0, measurer.complexity(context), selected_arms, exec_result, exec_cost, avg_accuracy, avg_cost, sum(gathered_reward)/len(gathered_reward)])
        pbar.set_description("Accuracy: {:.3f}, Average cost: {:.3f}".format(avg_accuracy, avg_cost))
        pbar.update(1)

    pbar.close()
    # print("Accuracy: ", total_correct/len(dataset))
    # print("Average cost: ", total_cost/len(dataset))
    with open(f"./log_new_{arg_cost_weighting}.txt", "w") as f:
        for l in log:
            f.write(", ".join([str(i) for i in l]) + "\n")

    print(f"**** Finished processing cost weighting: {arg_cost_weighting} ****")
  

if __name__ == "__main__":
    main()
