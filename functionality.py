from model import model_a1, model_a2, model_a3, model_b1, model_b2

def functionA(input, cost):
    if cost == 0:
        return model_a1(input), 10
    elif cost == 1:
        return model_a2(input), 30
    elif cost == 2:
        return model_a3(input), 50
    else:
        raise ValueError("Invalid cost")
    
def functionB(input, cost):
    if cost == 0:
        return model_b1(input), 10
    elif cost == 1:
        return model_b2(input), 50
    else:
        raise ValueError("Invalid cost")
