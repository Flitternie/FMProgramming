import ast
import inspect
import astor
from vipergpt.image_patch import ImagePatch
from routing import Router

routing_options = {
    'find': [30, 90],
    'exists': [30, 90],
    'verify_property': [182, 3770],
    'simple_query': [182, 3770]
}

class RoutingSystem:
    def __init__(self, func, source, cost_weighting):
        self.func = func
        self.source = source
        self.cost_weighting = cost_weighting
        self.function_calls = self.analyze_user_program()
        self.initialize(self.function_calls)
    
    def initialize(self, function_calls):
        # Initialize routing decisions based on function calls in the user program
        self.routing_info = {call['identifier']: 0 for call in function_calls}  # Default routing to 0 (small model)
        self.router = Router(self.routing_info, routing_options, self.cost_weighting)
    
    def make_routing_decisions(self, input_image) -> dict:
        routing_decisions, idx = self.router.select(input_image)
        return routing_decisions, idx
        
        # For testing purposes, always return the small model
        # return {key: 0 for key in self.routing_info.keys()}, 0
        # For testing purposes, always return the large model
        # return {key: 1 for key in self.routing_info.keys()}, self.router.num_arms-1

    # Function to modify the AST of the user program to add routing arguments
    def routing(self, input, display=False):
        tree = ast.parse(self.source)
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
        }
        exec(compiled_code, exec_globals)
        execute_command = exec_globals['execute_command']
        return execute_command, routing_decisions, idx

    def update_router(self, input_image, routing_idx, reward):
        # Update the router based on the reward received
        self.router.update(input_image, routing_idx, reward)

    # Function to extract the method calls and queries from a user program
    def analyze_user_program(self):
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