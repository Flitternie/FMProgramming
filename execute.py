from functionality import functionA, functionB

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
        else:
            raise ValueError("Invalid function")
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
    for step in program:
        if step[0] == "functionA":
            routed_program.append(("functionA", selected_arms[0]))
        elif step[0] == "functionB":
            routed_program.append(("functionB", selected_arms[1]))
        else:
            raise ValueError("Invalid function")
    exec_result, exec_cost = compute(input, routed_program)
    return exec_result, exec_cost

