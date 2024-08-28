from model import model_a1, model_a2, model_a3, model_b1, model_b2, model_c1, model_c2
from model import module_costs

def functionA(input, cost):
    if cost == 0:
        return model_a1(input), module_costs['functionA'][0]
    elif cost == 1:
        return model_a2(input), module_costs['functionA'][1]
    elif cost == 2:
        return model_a3(input), module_costs['functionA'][2]
    else:
        raise ValueError("Invalid cost")
    
def functionB(input, cost):
    if cost == 0:
        return model_b1(input), module_costs['functionB'][0]
    elif cost == 1:
        return model_b2(input), module_costs['functionB'][1]
    else:
        raise ValueError("Invalid cost")
    
def functionC(input, cost):
    if cost == 0:
        return model_c1(input), module_costs['functionC'][0]
    elif cost == 1:
        return model_c2(input), module_costs['functionC'][1]
    else:
        raise ValueError("Invalid cost")
