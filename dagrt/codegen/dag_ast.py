"""Abstract syntax"""
from pymbolic.mapper import IdentityMapper
from pymbolic.primitives import Expression, LogicalNot
from dagrt.language import Nop


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


class NullASTNode(Expression):

    init_arg_names = ()

    def __getinitargs__(self):
        return ()

    mapper_method = "map_NullASTNode"


class StatementWrapper(Expression):
    """
    .. attribute: statement
    """

    init_arg_names = ("statement",)

    def __init__(self, statement):
        self.statement = statement

    def __getinitargs__(self):
        return self.statement,

    mapper_method = "map_StatementWrapper"


def get_statements_in_ast(ast):
    """
    Return a generator that yields the statements in the AST in linear order.
    """

    if isinstance(ast, StatementWrapper):
        yield ast.statement
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
        for inst in get_statements_in_ast(child):
            yield inst


def create_ast_from_phase(code, phase_name):
    """
    Return an AST representation of the statements corresponding to the phase
    named `phase` as found in the :class:`DAGCode` instance `code`.
    """

    phase = code.phases[phase_name]

    # Construct a topological order of the statements.
    stack = []
    statement_map = dict((inst.id, inst) for inst in phase.statements)
    visiting = set()
    visited = set()
    topological_order = []

    # TODO: Clump nodes together in the topological order based on conditionals.

    stack.extend(sorted(phase.depends_on))
    while stack:
        statement = stack[-1]
        if statement in visited:
            if statement in visiting:
                visiting.remove(statement)
                topological_order.append(statement)
            stack.pop()
        else:
            visited.add(statement)
            visiting.add(statement)
            stack.extend(
                    sorted(statement_map[statement].depends_on))

    # Convert the topological order to an AST.
    main_block = []

    from pymbolic.primitives import LogicalAnd

    for statement in map(statement_map.__getitem__, topological_order):
        if isinstance(statement, Nop):
            continue

        # Statements become AST nodes. An unconditional statement is wrapped
        # into an StatementWrapper, while conditional statements are wrapped
        # using IfThens.

        if isinstance(statement.condition, LogicalAnd):
            # LogicalAnd(c1, c2, ...) => IfThen(c1, IfThen(c2, ...))
            conditions = reversed(statement.condition.children)
            inst = IfThenElse(next(conditions),
                              StatementWrapper(statement.copy(condition=True)),
                              NullASTNode())
            for next_cond in conditions:
                inst = IfThenElse(next_cond, inst, NullASTNode())
            main_block.append(inst)

        elif statement.condition is not True:
            main_block.append(IfThenElse(statement.condition,
                StatementWrapper(statement.copy(condition=True)),
                NullASTNode()))

        else:
            main_block.append(StatementWrapper(statement))

    ast = Block(*main_block)

    return simplify_ast(ast)


def simplify_ast(ast):
    """Return an optimized copy of the AST `ast`."""
    from six.moves import reduce

    def apply_pass(ast, pass_):
        return pass_(ast)

    passes = (
        ASTPreSimplifyMapper(),
        ASTSimplifyMapper(),
        ASTPostSimplifyMapper(),
    )

    return reduce(apply_pass, passes, ast)


class ASTIdentityMapper(IdentityMapper):

    def map_IfThenElse(self, expr):
        return type(expr)(self.rec(expr.condition), self.rec(expr.then),
                          self.rec(expr.else_))

    def map_IfThen(self, expr):
        return type(expr)(self.rec(expr.condition), self.rec(expr.then))

    def map_Block(self, expr):
        return type(expr)(*[self.rec(child) for child in expr.children])

    def map_NullASTNode(self, expr):
        return type(expr)()

    def map_StatementWrapper(self, expr):
        return type(expr)(expr.statement)


class ASTPreSimplifyMapper(ASTIdentityMapper):

    def map_IfThen(self, expr):
        return IfThenElse(expr.condition, self.rec(expr.then), NullASTNode())


class ASTPostSimplifyMapper(ASTIdentityMapper):

    def __call__(self, ast):
        ast = self.rec(ast)
        if isinstance(ast, NullASTNode):
            return Block()
        return ast

    def map_IfThenElse(self, expr):
        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)
        is_then_null = isinstance(then, NullASTNode)
        is_else_null = isinstance(else_, NullASTNode)
        if is_then_null and is_else_null:
            return NullASTNode()
        elif is_then_null:
            return IfThen(LogicalNot(expr.condition), else_)
        elif is_else_null:
            return IfThen(expr.condition, then)
        else:
            return IfThenElse(expr.condition, then, else_)

    def map_Block(self, expr):
        new_children = []
        for child in expr.children:
            child = self.rec(child)
            if isinstance(child, NullASTNode):
                continue
            new_children.append(child)
        if len(new_children) == 0:
            return NullASTNode()
        elif len(new_children) == 1:
            return new_children[0]
        else:
            return Block(*new_children)

    def map_StatementWrapper(self, expr):
        return StatementWrapper(expr.statement)


class ASTSimplifyMapper(ASTIdentityMapper):

    def map_IfThenElse(self, expr):
        if expr.condition is True:
            return self.rec(expr.then)
        elif expr.condition is False:
            return self.rec(expr.else_)

        condition = expr.condition
        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)

        # Simplify the condition.
        while isinstance(condition, LogicalNot):
            condition = condition.child
            then, else_ = (else_, then)

        # Simplify the children, if they have the same condition.
        if isinstance(then, IfThenElse) and condition == then.condition:
            then = then.then
        if isinstance(else_, IfThenElse) and condition == else_.condition:
            else_ = else_.else_

        return IfThenElse(condition, then, else_)

    def map_Block(self, expr):
        from collections import deque
        children_queue = deque(self.rec(child) for child in expr.children)

        if not children_queue:
            return expr

        children = []

        def flat_Block(*nodes):
            result = []
            for node in nodes:
                if isinstance(node, NullASTNode):
                    continue
                if isinstance(node, Block):
                    result.extend(node.children)
                else:
                    result.append(node)
            return Block(*result)

        # current_child is the current AST node that is being worked on.
        current_child = children_queue.popleft()
        while isinstance(current_child, NullASTNode):
            current_child = children_queue.popleft()

        while children_queue:
            next_child = children_queue.popleft()

            if isinstance(next_child, NullASTNode):
                continue

            # Expand any inner Blocks.
            if isinstance(next_child, Block):
                children_queue.extendleft(next_child.children)
                continue

            # Merge adjacent conditionals.
            if isinstance(current_child, IfThenElse) \
                    and isinstance(next_child, IfThenElse) \
                    and current_child.condition == next_child.condition:
                current_child = \
                    IfThenElse(current_child.condition,
                               flat_Block(current_child.then, next_child.then),
                               flat_Block(current_child.else_, next_child.else_))
                continue

            children.append(current_child)
            current_child = next_child

        children.append(current_child)

        if len(children) == 1:
            return children[0]
        return Block(*children)
