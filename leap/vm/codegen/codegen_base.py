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
from .ir import AssignInst, JumpInst, BranchInst, ReturnInst
from leap.vm.language import AssignExpression, AssignNorm, AssignRHS
from leap.vm.utils import TODO

from pytools import RecordWithoutPickling, memoize_method


class NewTimeIntegratorCode(RecordWithoutPickling):
    """
    A TimeIntegratorCode with staging support. This will eventually replace
    TimeIntegratorCode.

    .. attribute:: instructions

        is a list of Instruction instances, in no particular
        order

    .. attribute:: stages

        is a map from stage names to lists of ids corresponding to
        execution dependencies

    .. attribute:: initial_stage

        the name of the starting stage

    .. attribute:: step_before_fail

        is a boolean that indicates whether the described
        method may generate state updates for a time step it later decides
        to fail
    """

    @classmethod
    def from_old(cls, code):
        stages = {}
        stages['initialization'] = code.initialization_dep_on
        stages['primary'] = code.step_dep_on
        return cls(code.instructions, stages, 'initialization',
                   code.step_before_fail)

    def __init__(self, instructions, stages, initial_stage, step_before_fail):
        RecordWithoutPickling.__init__(self, instructions=instructions,
                                       stages=stages,
                                       initial_stage=initial_stage,
                                       step_before_fail=step_before_fail)


class StructuredCodeGenerator(object):
    """Code generation for structured languages"""

    def __call__(self, code):
        """Return *code* as a generated code string."""
        raise NotImplementedError()

    def lower_node(self, node):
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
            self.emit_if_end()
            self.emit_else_begin()
            self.lower_node(node.else_node)
            self.emit_else_end()
        elif isinstance(node, BlockNode):
            for node_item in node.node_list:
                self.lower_node(node_item)
        elif isinstance(node, UnstructuredIntervalNode):
            raise TODO(
                    'Implement lowering for unstructured intervals')

    def lower_inst(self, inst):
        if isinstance(inst, AssignInst):
            assignment = inst.assignment
            if isinstance(assignment, tuple):
                self.emit_assign_expr(assignment[0], assignment[1])
            elif isinstance(assignment, AssignExpression):
                self.emit_assign_expr(assignment.assignee,
                                      assignment.expression)
            elif isinstance(assignment, AssignNorm):
                self.emit_assign_norm(assignment.assignee,
                                      assignment.expression, assignment.p)
            elif isinstance(assignment, AssignRHS):
                # Lower each parallel assignment sequentially, for now.
                rhs = assignment.component_id
                time = assignment.t
                args = assignment.rhs_arguments
                for index, assignee in enumerate(assignment.assignees):
                    self.emit_assign_rhs(assignee, rhs, time, args[index])
            else:
                raise TODO('Lower all assignment types')
        elif isinstance(inst, JumpInst):
            pass
        elif isinstance(inst, BranchInst):
            self.emit_if_begin(inst.condition)
        elif isinstance(inst, ReturnInst):
            self.emit_return(inst.expression)

    # Emit routines (to be implemented by subclass)

    def begin_emit(self):
        raise NotImplementedError()

    def finish_emit(self):
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

    def emit_else_end(self):
        raise NotImplementedError()

    def emit_assign_expr(self, name, expr):
        raise NotImplementedError()

    def emit_assign_norm(self, name, expr, p):
        raise NotImplementedError()

    def emit_assign_rhs(self, name, rhs, time, arg):
        raise NotImplementedError()

    def emit_return(self, expr):
        raise NotImplementedError()
