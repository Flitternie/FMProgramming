import ast
import inspect
import astor
from execution.image_patch import ImagePatch, distance
from routing import Router, StructuredRouter

routing_options = {
    'find': [30, 90],
    'exists': [30, 90],
    'verify_property': [182, 3770],
    'simple_query': [182, 3770]
}

class RoutingSystem:
    def __init__(self, func, source, cost_weighting, struct=True):
        '''
        This class initializes the routing system for the user program.

        Args:
            func: Function to execute the user program
            source: Source code of the user program
            cost_weighting: Cost weighting parameter for the router

        Methods:
            initialize: Initializes the routing decisions based on the function calls in the user program
            make_routing_decisions: Makes routing decisions based on the input image
            routing: Modifies the AST of the user program to add routing arguments
            update_router: Updates the router based on the reward received
            analyze_user_program: Extracts the method calls and queries from a user program
        
        '''
        self.func = func
        self.source = source
        self.cost_weighting = cost_weighting
        self.function_calls = self.analyze_user_program()
        self.initialize(self.function_calls, struct)
    
    def initialize(self, function_calls, struct):
        # Initialize routing decisions based on function calls in the user program
        '''
        This function initializes the routing decisions based on the function calls in the user program.

        Args:
            function_calls: List of dictionaries containing the function calls and queries

        '''
        self.routing_info = {call['identifier']: 0 for call in function_calls}  # Default routing to 0 (small model)
        if struct:
            self.router = StructuredRouter(self.routing_info, routing_options, self.cost_weighting)
        else:
            self.router = Router(self.routing_info, routing_options, self.cost_weighting)
    
    def make_routing_decisions(self, input_image) -> dict:
        '''
        This function makes routing decisions based on the input image.

        Args:
            input_image: ImagePatch object
        
        Returns:
            routing_decisions: Dictionary containing the routing decisions for each function call
            idx: Index of the arm or the group of arms selected by the router

        '''
        routing_decisions, idx = self.router.select(input_image)
        return routing_decisions, idx

    def routing(self, input, config=None, display=False):
        '''
        This function modifies the AST of the user program to add routing arguments.

        Args:
            input: ImagePatch object
            config: None for dynamic routing, or 0 for static routing with the smallest model configuration, or 1 for the largest model configuration
            display: Boolean to display the modified AST

        Returns:
            execute_command: Function to execute the modified user program
            routing_decisions: Dictionary containing the routing decisions for each function call
            idx: Index of the arm selected by the router
        
        '''
        tree = ast.parse(self.source)
        if config is not None:
            assert config in [0, 1], "Invalid configuration"
            routing_decision = -1 if config == 1 else 0
            routing_decisions = {key: routing_decision for key in self.routing_info.keys()}
            idx = self.router.num_arms - 1 if config == 1 else 0
        else:
            routing_decisions, idx = self.make_routing_decisions(input)

        class RoutingArgumentTransformer(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    instance_name = node.func.value.id
                    method_name = node.func.attr
                
                    if method_name in routing_options.keys():
                        # Generate an identifier to match against routing info
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            query = node.args[0].s
                            identifier = f"{method_name}('{query}')"
                            routing_value = int(routing_decisions[identifier])

                            # Add the routing argument to the function call
                            node.keywords.append(ast.keyword(arg='routing', value=ast.Constant(value=routing_value)))

                return self.generic_visit(node)

        transformer = RoutingArgumentTransformer()
        modified_tree = transformer.visit(tree)
        if display:
            print(astor.to_source(modified_tree))
        ast.fix_missing_locations(modified_tree)
        compiled_code = compile(modified_tree, filename="<ast>", mode="exec")

        # Adding imports and necessary globals to ensure everything is available during execution
        exec_globals = {
            '__builtins__': __builtins__,  # Provide access to built-ins
            'ImagePatch': ImagePatch,      # Ensure ImagePatch is available during execution
            'distance': distance,          # Ensure distance function is available during execution
        }
        exec(compiled_code, exec_globals)
        execute_command = exec_globals['execute_command']
        return execute_command, routing_decisions, idx

    def update_router(self, input_image, routing_idx, reward):
        '''
        This function updates the router based on the reward received.

        Args:
            input_image: ImagePatch object
            routing_idx: Index of the arm selected by the router
            reward: Reward received
        
        '''
        self.router.update(input_image, routing_idx, reward)

    def analyze_user_program(self):
        '''
        This function extracts the method calls and queries from a user program.

        Returns:
            function_calls: List of dictionaries containing the function calls and queries

        '''
        self.source = inspect.getsource(self.source) if not isinstance(self.source, str) else self.source
        tree = ast.parse(self.source)
        function_calls = []

        class MethodCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    # Extract the function call details
                    method_name = node.func.attr
                    instance_name = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                    # Get arguments, specifically for calls like find, vqa, etc.
                    if method_name in routing_options.keys():
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            query = node.args[0].s
                            identifier = f"{method_name}('{query}')"
                            function_calls.append({
                                'method_name': method_name,
                                'query': query,
                                'identifier': identifier
                            })
                self.generic_visit(node)

        MethodCallVisitor().visit(tree)
        return function_calls