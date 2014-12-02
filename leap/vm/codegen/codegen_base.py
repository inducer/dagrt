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

from .structured_ir import SingleNode, IfThenNode, IfThenElseNode, BlockNode, \
    UnstructuredIntervalNode
from .ir import AssignInst, JumpInst, BranchInst, ReturnInst, YieldStateInst, \
    RaiseInst, FailStepInst
from leap.vm.language import AssignExpression
from leap.vm.utils import TODO


# {{{ structured code generator

class StructuredCodeGenerator(object):
    """Code generation for structured languages"""

    def __call__(self, code):
        """Return *code* as a generated code string."""
        raise NotImplementedError()

    def lower_node(self, node):
        """Emit the code in the ControlTree node.

        :arg node: A :class:`ControlTree` node to lower
        """
        if isinstance(node, SingleNode):
            for inst in node.basic_block.code:
                self.lower_inst(inst)
        elif isinstance(node, IfThenNode):
            self.lower_node(node.if_node)
            self.lower_node(node.then_node)
            self.emit_if_end()
        elif isinstance(node, IfThenElseNode):
            self.lower_node(node.if_node)
            self.lower_node(node.then_node)
            self.emit_else_begin()
            self.lower_node(node.else_node)
            self.emit_if_end()
        elif isinstance(node, BlockNode):
            for node_item in node.node_list:
                self.lower_node(node_item)
        elif isinstance(node, UnstructuredIntervalNode):
            raise TODO(
                    'Implement lowering for unstructured intervals')

    def lower_inst(self, inst):
        """Emit the code for an instruction."""
        if isinstance(inst, AssignInst):
            assignment = inst.assignment
            if isinstance(assignment, tuple):
                self.emit_assign_expr(assignment[0], assignment[1])
            elif isinstance(assignment, AssignExpression):
                self.emit_assign_expr(assignment.assignee,
                                      assignment.expression)
            else:
                raise ValueError("unrecognized assignment type '%s'"
                        % type(inst.assignment).__name__)

        elif isinstance(inst, JumpInst):
            pass
        elif isinstance(inst, BranchInst):
            self.emit_if_begin(inst.condition)
        elif isinstance(inst, ReturnInst):
            self.emit_return()
        elif isinstance(inst, YieldStateInst):
            self.emit_yield_state(inst)
        elif isinstance(inst, RaiseInst):
            leap_instruction = inst.instruction
            self.emit_raise(leap_instruction.error_condition,
                            leap_instruction.error_message)
        elif isinstance(inst, FailStepInst):
            self.emit_fail_step()

    # Emit routines (to be implemented by subclass)

    def begin_emit(self, dag):
        raise NotImplementedError()

    def finish_emit(self, dag):
        raise NotImplementedError()

    def emit_def_begin(self, name):
        raise NotImplementedError()

    def emit_def_end(self):
        raise NotImplementedError()

    def emit_while_loop_begin(self, expr):
        raise NotImplementedError()

    def emit_while_loop_end(self):
        raise NotImplementedError()

    def emit_if_begin(self, expr):
        raise NotImplementedError()

    def emit_if_end(self):
        raise NotImplementedError()

    def emit_else_begin(self):
        raise NotImplementedError()

    def emit_assign_expr(self, name, expr):
        raise NotImplementedError()

    def emit_return(self):
        raise NotImplementedError()

    def emit_yield_state(self, int):
        raise NotImplementedError()

    def emit_raise(self, error_condition, error_message):
        raise NotImplementedError()

    def emit_fail_step(self):
        raise NotImplementedError()

# }}}
