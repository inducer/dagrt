"""Base class for code generators"""

__copyright__ = "Copyright (C) 2014 Matt Wala"

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

from .ast_ import Block, IfThen, IfThenElse, InstructionWrapper
from leap.vm.language import AssignExpression, YieldState, Raise, FailStep, \
    StateTransition, Nop


# {{{ structured code generator

class StructuredCodeGenerator(object):
    """Code generation for structured languages"""

    def __call__(self, code):
        """Return *code* as a generated code string."""
        raise NotImplementedError()

    def lower_ast(self, ast):
        self.lower_node(ast)
        self.emit_return()

    def lower_node(self, node):
        """
        :arg node: An AST node to lower
        """
        if isinstance(node, InstructionWrapper):
            self.lower_inst(node.instruction)

        elif isinstance(node, IfThen):
            self.emit_if_begin(node.condition)
            self.lower_node(node.then)
            self.emit_if_end()

        elif isinstance(node, IfThenElse):
            self.emit_if_begin(node.condition)
            self.lower_node(node.then)
            self.emit_else_begin()
            self.lower_node(node.else_)
            self.emit_if_end()

        elif isinstance(node, Block):
            for child in node.children:
                self.lower_node(child)

        else:
            raise ValueError("Unrecognized node type {type}".format(
                type=type(node).__name__))

    def lower_inst(self, inst):
        """Emit the code for an instruction."""

        if isinstance(inst, AssignExpression):
            self.emit_assign_expr(inst.assignee, inst.expression)

        elif isinstance(inst, YieldState):
            self.emit_yield_state(inst.component_id, inst.expression,
                                  inst.time, inst.time_id)

        elif isinstance(inst, Raise):
            self.emit_raise(inst.error_condition, inst.error_message)

        elif isinstance(inst, FailStep):
            self.emit_fail_step()

        elif isinstance(inst, StateTransition):
            self.emit_state_transition(inst.next_state)

        elif isinstance(inst, Nop):
            pass

        else:
            raise ValueError("Unrecognized instruction type {type}".format(
                type=type(inst).__name__))

    # Emit routines (to be implemented by subclass)

    def begin_emit(self, dag):
        raise NotImplementedError()

    def finish_emit(self, dag):
        raise NotImplementedError()

    def emit_def_begin(self, name):
        raise NotImplementedError()

    def emit_def_end(self):
        raise NotImplementedError()

    def emit_if_begin(self, expr):
        raise NotImplementedError()

    def emit_if_end(self):
        raise NotImplementedError()

    def emit_else_begin(self):
        raise NotImplementedError()

    def emit_assign_expr(self, name, expr):
        raise NotImplementedError()

    def emit_yield_state(self, component_id, expression, time, time_id):
        raise NotImplementedError()

    def emit_return(self):
        raise NotImplementedError()

    def emit_raise(self, error_condition, error_message):
        raise NotImplementedError()

    def emit_fail_step(self):
        raise NotImplementedError()

    def emit_state_transition(self, next_state):
        raise NotImplementedError()

# }}}
