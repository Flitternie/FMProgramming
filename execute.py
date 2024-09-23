from model import *

'''
This file contains the code to execute the program and route it to the specific arms.
'''

func_As = [model_a1, model_a2, model_a3]
func_Bs = [model_b1, model_b2]
func_Cs = [model_c1, model_c2, model_c3, model_c4]

def compute(input, program):
    info = [input]
    costs = []
    for subprogram in program:
        function, cost = subprogram
        if function == "functionA":
            output, cost = functionA(info, cost)
            info.append(output)
            costs.append(cost)
        elif function == "functionB":
            output, cost = functionB(info, cost)
            info.append(output)
            costs.append(cost)
        elif function == "functionC":
            output, cost = functionC(info, cost)
            info.append(output)
            costs.append(cost)
    return info[-1], sum(costs)

def idx_to_arms(idx, modules):
    arms = []
    for _, num_arms in modules.items():
        arms.append(idx % num_arms)
        idx = idx // num_arms
    return arms

def routing(input, program, selected_arms):
    routed_program = []
    assert len(program) == len(selected_arms)
    for step, arm in zip(program, selected_arms):
        if step[0] == "functionA":
            routed_program.append(("functionA", arm))
        elif step[0] == "functionB":
            routed_program.append(("functionB", arm))
        elif step[0] == "functionC":
            routed_program.append(("functionC", arm))
        else:
            raise ValueError("Invalid function")
    exec_result, exec_cost = compute(input, routed_program)
    return exec_result, exec_cost

def functionA(input, cost):
    try:
        return func_As[cost](input), module_costs['functionA'][cost]
    except Exception as e:
        print("Invalid cost: {}".format(cost))
        raise e
    
def functionB(input, cost):
    try:
        return func_Bs[cost](input), module_costs['functionB'][cost]
    except Exception as e:
        print("Invalid cost: {}".format(cost))
        raise e
    
def functionC(input, cost):
    try:
        return func_Cs[cost](input), module_costs['functionC'][cost]
    except Exception as e:
        print("Invalid cost: {}".format(cost))
        raise e
