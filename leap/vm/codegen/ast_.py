"""Abstract syntax"""

__copyright__ = "Copyright (C) 2015 Matt Wala"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from pymbolic.mapper import Collector, IdentityMapper
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.mapper.unifier import UnidirectionalUnifier
from pymbolic.primitives import Expression, LogicalNot
from leap.vm.language import Nop


class IfThen(Expression):
    """
    .. attribute: condition
    .. attribute: then
    """

    init_arg_names = ("condition", "then")

    def __init__(self, condition, then):
        self.condition = condition
        self.then = then

    def __getinitargs__(self):
        return self.condition, self.then

    mapper_method = "map_IfThen"


class IfThenElse(Expression):
    """
    .. attribute: condition
    .. attribute: then
    .. attribute: else_
    """

    init_args_names = ("condition", "then", "else_")

    def __init__(self, condition, then, else_):
        self.condition = condition
        self.then = then
        self.else_ = else_

    def __getinitargs__(self):
        return self.condition, self.then, self.else_

    mapper_method = "map_IfThenElse"


class Block(Expression):
    """
    .. attribute: children
    """

    init_arg_names = ("children",)

    def __init__(self, *children):
        self.children = children

    def __getinitargs__(self):
        return self.children

    mapper_method = "map_Block"


class InstructionWrapper(Expression):
    """
    .. attribute: instruction
    """

    init_arg_names = ("instruction",)

    def __init__(self, instruction):
        self.instruction = instruction

    def __getinitargs__(self):
        return self.instruction,

    mapper_method = "map_InstructionWrapper"


class _ASTCombinerMixin(object):

    def map_IfThen(self, expr):
        return self.combine([self.rec(expr.condition), self.rec(expr.then)])

    def map_IfThenElse(self, expr):
        return self.combine([self.rec(expr.condition), self.rec(expr.then),
                             self.rec(expr.else_)])

    def map_Block(self, expr):
        return self.combine([self.rec(child) for child in expr.children])


class _ExtendedDependencyMapper(_ASTCombinerMixin, DependencyMapper):
    pass


class _ASTDefinedVariablesCollector(_ASTCombinerMixin, Collector):

    def map_InstructionWrapper(self, expr):
        return expr.instruction.get_assignees()


class _ASTMatcher(UnidirectionalUnifier):
    """
    Extend UnidirectionalUnifier to handle matching of AST fragments (along with
    LogicalNot).
    """

    def map_logical_not(self, expr, other, urecs):
        if not isinstance(other, type(expr)):
            return []
        return self.rec(expr.child, other.child, urecs)

    def map_IfThen(self, expr, other, urecs):
        if not isinstance(other, type(expr)):
            return []
        urecs = self.rec(expr.condition, other.condition, urecs)
        return self.rec(expr.then, other.then, urecs)

    def map_IfThenElse(self, expr, other, urecs):
        if not isinstance(other, type(expr)):
            return []
        urecs = self.rec(expr.condition, other.condition, urecs)
        urecs = self.rec(expr.then, other.then, urecs)
        return self.rec(expr.else_, other.else_, urecs)

    def map_Block(self, expr, other, urecs):
        if not isinstance(other, type(expr)):
            return []

        if len(expr.children) != len(other.children):
            return []

        for child, other_child in zip(expr.children, other.children):
            urecs = self.rec(child, other_child, urecs)

        return urecs


def get_all_variables_in_ast(ast):
    """Return the names of all the variables in the AST fragment `ast`."""
    # TODO: Cache results / copy caches over when copying AST fragments
    return _ExtendedDependencyMapper()(ast)


def get_instructions_in_ast(ast):
    """
    Return a generator that yields the instructions in the AST in linear order.
    """

    if isinstance(ast, InstructionWrapper):
        yield ast.instruction
        return

    if isinstance(ast, IfThen):
        children = (ast.then,)
    elif isinstance(ast, IfThenElse):
        children = (ast.then, ast.else_)
    elif isinstance(ast, Block):
        children = ast.children
    else:
        raise ValueError("Unknown node type: {}".format(ast.__class__.__name__))

    for child in children:
        for inst in get_instructions_in_ast(child):
            yield inst


def redefines(ast, expression):
    """
    Return whether any of the variables in `expression` are redefined
    in the AST fragment `ast`.
    """

    from leap.vm.utils import get_variables
    variables = get_variables(expression)
    # TODO: Cache results / copy caches over when copying AST fragments
    redefined_variables = _ASTDefinedVariablesCollector()(ast)

    return bool(variables & redefined_variables)


def create_ast_from_state(code, state, simplify=True):
    """
    Return an AST representation of the instructions corresponding to the state
    named `state` as found in the :class:`TimeIntegratorCode` instance `code`.
    """

    # Construct a topological order of the instructions.
    stack = []
    instruction_map = dict((inst.id, inst) for inst in code.instructions)
    visiting = set()
    visited = set()
    topological_order = []

    # TODO: Clump nodes together in the topological order based on conditionals.

    stack.extend(list(code.states[state].depends_on))
    while stack:
        instruction = stack[-1]
        if instruction in visited:
            if instruction in visiting:
                visiting.remove(instruction)
                topological_order.append(instruction)
            stack.pop()
        else:
            visited.add(instruction)
            visiting.add(instruction)
            for dependency in instruction_map[instruction].depends_on:
                stack.append(dependency)

    # Convert the topological order to an AST.
    main_block = []

    from pymbolic.primitives import LogicalAnd

    for instruction in map(instruction_map.__getitem__, topological_order):
        if simplify and isinstance(instruction, Nop):
            continue

        # Instructions become AST nodes. An unconditional instruction is wrapped
        # into an InstructionWrapper, while conditional instructions are wrapped
        # using IfThens.

        if isinstance(instruction.condition, LogicalAnd):
            # LogicalAnd(c1, c2, ...) => IfThen(c1, IfThen(c2, ...))
            conditions = reversed(instruction.condition.children)
            inst = IfThen(next(conditions),
                          InstructionWrapper(instruction.copy(condition=True)))
            for next_cond in conditions:
                inst = IfThen(next_cond, inst)
            main_block.append(inst)

        elif instruction.condition is not True:
            main_block.append(IfThen(instruction.condition,
                              InstructionWrapper(instruction.copy(condition=True))))

        else:
            main_block.append(InstructionWrapper(instruction))

    ast = Block(*main_block)

    if simplify:
        ast = simplify_ast(ast)

    return ast


def simplify_ast(ast):
    """Return an optimized copy of the AST `ast`."""
    MAX_ITERS = 5
    mapper = _ASTSimplificationMapper()
    ast = mapper(ast)
    mapper_application_count = 1

    while mapper.changed and mapper_application_count < MAX_ITERS:
        # TODO: Implement the following optimizations:
        #
        # 1. Hoisting of <cond> into if statements: the frontend translates
        #
        #    with cb.if_(complex_condition):
        #        cb("x", 1)
        #
        # into
        #
        #    AssignExpression("<cond>var", complex_condition)
        #    AssignExpression("x", 1, condition=<cond>var).
        #
        # While this simplifies the subsequent code generation steps, the end
        # result is not as readable as it could be because of the temparary
        # assignment, so we should undo this at a late stage.
        #
        # 2. Dead code elimination: for cleaning up after (1) as well as for
        # generally improving readability of the code.
        ast = mapper(ast)
        mapper_application_count += 1

    return ast


def match_ast(expression, template):
    """
    Do pattern matching for AST fragments.

    :param expression: A :mod:`pymbolic` AST fragment
    :param template: A :mod`pymbolic` AST fragment whose variables will be matched
    with subexpressions of `expression`

    :return: A map from variable names to subexpressions, or `None` if no match
    was found
    """
    names = [var.name for var in get_all_variables_in_ast(template)]
    matcher = _ASTMatcher(names)
    records = matcher(template, expression)
    if not records:
        return None
    return dict((key.name, val) for key, val in records[0].equations)


def declare(*varnames):
    """
    Do a bulk declaration of :mod:`pymbolic` variables.
    """
    from pymbolic import var
    return [var(name) for name in varnames]


class _ASTSimplificationMapper(IdentityMapper):

    def __call__(self, ast):
        self.changed = False
        return self.rec(ast)

    def map_IfThen(self, expr):
        return IfThen(self.rec(expr.condition), self.rec(expr.then))

    def map_IfThenElse(self, expr):
        return IfThenElse(self.rec(expr.condition), self.rec(expr.then),
                          self.rec(expr.else_))

    def map_Block(self, expr):
        from collections import deque
        children_queue = deque(self.rec(child) for child in expr.children)

        if not children_queue:
            return expr

        children = []

        # current_child is the current AST node that is being worked on by the
        # algorithm.
        current_child = children_queue.popleft()

        # Variables for pattern matching
        p, t, tt, ttt = declare("p", "t", "tt", "ttt")

        def flat_Block(*nodes):
            result = []
            for node in nodes:
                if isinstance(node, Block):
                    result.extend(node.children)
                else:
                    result.append(node)
            return Block(*result)

        while children_queue:
            next_child = children_queue.popleft()

            # Expand any inner Blocks first.
            if isinstance(next_child, Block):
                children_queue.extendleft(next_child.children)
                self.changed = True
                continue

            child_pair = (current_child, next_child)

            # IfThen(p, t), IfThen(p, tt) => IfThen(p, Block(t, tt))
            match = match_ast(child_pair, (IfThen(p, t), IfThen(p, tt)))
            if match and not redefines(match["t"], match["p"]):
                current_child = IfThen(match["p"],
                                       flat_Block(match["t"], match["tt"]))
                self.changed = True
                continue

            # IfThen(p, t), IfThen(not p, tt) => IfThenElse(p, t, tt)
            match = match_ast(child_pair, (IfThen(p, t), IfThen(LogicalNot(p), tt)))
            if match and not redefines(match["t"], match["p"]):
                current_child = IfThenElse(match["p"], match["t"], match["tt"])
                self.changed = True
                continue

            # IfThen IfThenElse -> IfThenElse
            match = match_ast(child_pair, (IfThen(p, t), IfThenElse(p, tt, ttt)))
            if match and not redefines(match["t"], match["p"]):
                current_child = IfThenElse(match["p"],
                                           flat_Block(match["t"], match["tt"]),
                                           match["ttt"])
                self.changed = True
                continue

            # IfThenElse(p, t, tt), IfThen(not p, ttt)
            #  => IfThenElse(p, t, Block(tt, ttt))
            match = match_ast(child_pair, (IfThenElse(p, t, tt),
                                           IfThen(LogicalNot(p), ttt)))
            if match and not redefines(match["t"], match["p"]) and not \
                    redefines(match["tt"], match["p"]):
                current_child = IfThenElse(match["p"], match["t"],
                                           flat_Block(match["tt"], match["ttt"]))
                self.changed = True
                continue

            # No opportunity for combining the children was found.
            children.append(current_child)
            current_child = next_child

        children.append(current_child)

        if len(children) == 1:
            return children[0]
        return Block(*children)

    def map_InstructionWrapper(self, expr):
        return InstructionWrapper(expr.instruction)
