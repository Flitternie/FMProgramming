import ast
import inspect
import astor
import functools
import warnings

from execution.modules import get_cost_info
from execution.image_patch import ImagePatch, distance, to_numeric
from routing import Router, StructuredRouter

"""
This file contains the routing system backend that analyzes the user program and 
dynamically injects routing arguments at runtime. It also tracks function and method 
calls to monitor routing behavior and compute cost and execution metrics. 
"""

# ---------------------------------------------------------------------------- #
#                               Routing Utilities                              #
# ---------------------------------------------------------------------------- #

def get_routing_options():
    """
    Retrieve routing cost options for different operation types.

    Returns:
        dict: Mapping of operation names to their cost lists per model.
    """
    cost_info = get_cost_info()
    return {
        'find': cost_info["object_detection"],
        'exists': cost_info["object_detection"],
        'verify_property': cost_info["vqa"],
        'query': cost_info["vqa"],
        'llm_query': cost_info["llm"]
    }

# ---------------------------------------------------------------------------- #
#                           Tracking & Decorators                              #
# ---------------------------------------------------------------------------- #

tracked_methods = ["find", "exists", "verify_property", "query", "llm_query"]
tracked_functions = []

method_call_tracker = {}
function_call_tracker = {}
execution_trace = set()


def count_method_calls(func):
    """
    Decorator to count method calls with routing and log execution trace.

    Args:
        func (Callable): The method to decorate.

    Returns:
        Callable: Decorated method.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global method_call_tracker, execution_trace

        method_name = func.__name__
        routing_value = kwargs.get("routing", None)
        query = ", ".join([arg for arg in args if isinstance(arg, str)])
        key = (method_name, routing_value)

        method_call_tracker[key] = method_call_tracker.get(key, 0) + 1

        if query:
            execution_trace.add(f"{method_name}('{query}')")

        return func(*args, **kwargs)

    return wrapper


def count_function_calls(func):
    """
    Decorator to count function calls with routing and log execution trace.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global function_call_tracker, execution_trace

        function_name = func.__name__
        routing_value = kwargs.get("routing", None)
        query = ", ".join([arg for arg in args if isinstance(arg, str)])
        key = (function_name, routing_value)

        function_call_tracker[key] = function_call_tracker.get(key, 0) + 1

        if query:
            execution_trace.add(f"{function_name}('{query}')")

        return func(*args, **kwargs)

    return wrapper


def wrap_specific_methods(cls, method_names):
    """
    Apply method tracking decorator to selected methods in a class.

    Args:
        cls (type): Class to decorate.
        method_names (list): List of method names.
    """
    for method_name in method_names:
        if hasattr(cls, method_name):
            original_method = getattr(cls, method_name)
            if callable(original_method):
                setattr(cls, method_name, count_method_calls(original_method))


def wrap_specific_functions(function_list):
    """
    Apply function tracking decorator to selected functions.

    Args:
        function_list (list): List of function objects.
    """
    for func in function_list:
        if callable(func):
            setattr(func.__globals__, func.__name__, count_function_calls(func))


# Apply decorators
wrap_specific_methods(ImagePatch, tracked_methods)
wrap_specific_functions(tracked_functions)

# ---------------------------------------------------------------------------- #
#                          Routed Program Execution                            #
# ---------------------------------------------------------------------------- #

def execute_routed_program(routed_program, image):
    """
    Execute routed program and collect execution statistics.

    Args:
        routed_program (Callable): The routed function.
        image (ImagePatch): Input image.

    Returns:
        tuple: (output, execution_counter, execution_trace)
    """
    global method_call_tracker, function_call_tracker, execution_trace

    method_call_tracker.clear()
    function_call_tracker.clear()
    execution_trace.clear()

    try:
        output = routed_program(image)
    except Exception as e:
        warnings.warn(f"Error executing the routed program: {e}")
        output = e

    execution_counter = []
    for (method_name, routing_value), count in method_call_tracker.items():
        execution_counter.append((method_name, routing_value, count))
    for (function_name, routing_value), count in function_call_tracker.items():
        execution_counter.append((function_name, routing_value, count))

    return output, execution_counter, execution_trace


def check_execution(execution_trace, function_calls):
    """
    Check if expected method+query combinations were executed.

    Args:
        execution_trace (set): Set of executed identifiers.
        function_calls (list): List of expected identifiers.

    Returns:
        list: List of 0/1 values.
    """
    return [1 if call in execution_trace else 0 for call in function_calls]


def execution_cost(execution_counter):
    """
    Compute total execution cost from function call statistics.

    Args:
        execution_counter (list): List of (function, routing, count).

    Returns:
        float: Total execution cost.
    """
    cost = 0
    routing_options = get_routing_options()
    for function, routing, count in execution_counter:
        if routing is not None:
            cost += count * routing_options[function][routing]
    return cost

# ---------------------------------------------------------------------------- #
#                                Routing System                                #
# ---------------------------------------------------------------------------- #

class RoutingSystem:
    """
    RoutingSystem dynamically inserts routing arguments into the user program 
    based on model cost configurations and learned strategies.

    Supports reinforcement learning and structured routing.

    Args:
        execute_command (Callable): Executable user function.
        source (str | Callable): Source code or function definition.
        cost_weighting (float): Weight for cost during routing.
        config (str): Routing algorithm ('struct_reinforce', 'reinforce', 'bandit').
    """

    def __init__(self, execute_command, source, cost_weighting, config="struct_reinforce", **kwargs):
        self.func = execute_command
        self.source = source
        self.cost_weighting = cost_weighting
        self.routing_options = get_routing_options()
        self.function_calls = self.analyze_user_program()
        self.initialize(self.function_calls, config, **kwargs)

    def initialize(self, function_calls, config, **kwargs):
        """
        Initialize router based on function calls and config strategy.
        """
        self.routing_info = {call['identifier']: 0 for call in function_calls}
        if config == "struct_reinforce":
            self.router = StructuredRouter(self.routing_info, self.routing_options, self.cost_weighting, **kwargs)
        elif config in ("reinforce", "bandit"):
            self.router = Router(self.routing_info, self.routing_options, self.cost_weighting, config, **kwargs)
        else:
            raise ValueError("Invalid configuration")

    def make_routing_decisions(self, input_image):
        """
        Make routing decisions based on current router state.

        Returns:
            tuple: (routing_decisions, selected_arm_index)
        """
        return self.router.select(input_image)

    def routing(self, input, config=None, display=False):
        """
        Modify AST of user program with routing args and return executable function.

        Args:
            input (ImagePatch): Input image.
            config (int | None): 0 for small, 1 for large, None for dynamic.
            display (bool): If True, print modified source.

        Returns:
            tuple: (execute_command, routing_decisions, routing_idx)
        """
        tree = ast.parse(self.source)

        if config is not None:
            assert config in [0, 1]
            routing_decision = -1 if config == 1 else 0
            routing_decisions = {k: routing_decision for k in self.routing_info}
            idx = self.router.num_arms - 1 if config == 1 else 0
        else:
            routing_decisions, idx = self.make_routing_decisions(input)

        routing_options = self.routing_options
        class RoutingArgumentTransformer(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name in routing_options:
                        query = ", ".join([arg.s for arg in node.args if isinstance(arg, ast.Str)])
                        identifier = f"{method_name}('{query}')"
                        routing_value = int(routing_decisions.get(identifier, 0))
                        node.keywords.append(ast.keyword(arg='routing', value=ast.Constant(value=routing_value)))
                return self.generic_visit(node)

        transformer = RoutingArgumentTransformer()
        modified_tree = transformer.visit(tree)
        if display:
            print(astor.to_source(modified_tree))
        ast.fix_missing_locations(modified_tree)

        exec_globals = {
            '__builtins__': __builtins__,
            'ImagePatch': ImagePatch,
            'distance': distance,
            'to_numeric': to_numeric
        }

        compiled_code = compile(modified_tree, filename="<ast>", mode="exec")
        exec(compiled_code, exec_globals)
        execute_command = exec_globals['execute_command']

        return execute_command, routing_decisions, idx

    def update_router(self, input_image, routing_idx, reward, reward_mapping):
        """
        Update router using received reward.
        """
        self.router.update(input_image, routing_idx, reward, reward_mapping)

    def analyze_user_program(self):
        """
        Analyze source code and extract identifiable function calls.

        Returns:
            list: List of method call dictionaries.
        """
        self.source = inspect.getsource(self.source) if not isinstance(self.source, str) else self.source
        tree = ast.parse(self.source)
        function_calls = []
        routing_options = self.routing_options

        class MethodCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name in routing_options:
                        query = ", ".join([arg.s for arg in node.args if isinstance(arg, ast.Str)])
                        identifier = f"{method_name}('{query}')"
                        function_calls.append({
                            'method_name': method_name,
                            'query': query,
                            'identifier': identifier
                        })
                self.generic_visit(node)

        MethodCallVisitor().visit(tree)
        return function_calls
