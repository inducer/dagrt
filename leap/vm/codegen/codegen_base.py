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

from .analysis import InstructionDAGVerifier
from .dag2ir import InstructionDAGExtractor, ControlFlowGraphAssembler
from .optimization import Optimizer
from warnings import warn

from pytools import RecordWithoutPickling, memoize_method

from leap.vm.language import ReturnState


class NewTimeIntegratorCode(RecordWithoutPickling):
    """A TimeIntegratorCode with staging support."""
    
    @classmethod
    def from_old(cls, code):
        stages = {}
        stages['initialization'] = code.initialization_dep_on
        stages['primary'] = code.step_dep_on
        # Insert a new attribute to the Return instructions that indicates the
        # name of the next stage.
        for stage, dependencies in stages.iteritems():
            stack = list(dependencies)
            visited = set()
            while stack:
                top = stack.pop()
                if top not in visited:
                    visited.add(top)
                    top_inst = code.id_to_insn[top]
                    stack.extend(list(top_inst.depends_on))
            for inst_id in visited:
                inst = code.id_to_insn[inst_id]
                if isinstance(inst, ReturnState):
                    # Add a new field to inst.
                    if stage == 'initialization':
                        inst.next_stage = 'primary'
                    elif stage == 'primary':
                        inst.next_stage = 'primary'
        return cls(code.instructions, stages, 'initialization',
                   code.step_before_fail)

    def __init__(self, instructions, stages, initial_stage, step_before_fail):
        """
        - instructions is a list of Instruction instances, in no particular
          order
        - stages is a map from stage names to lists of ids corresponding to
          execution dependencies
        - initial_state is the name of the starting stage
        - step_before_fail is a boolean that indicates whether the described
          method may generate state updates for a time step it later decides
          to fail
        """
        RecordWithoutPickling.__init__(self, instructions=instructions,
                                       stages=stages,
                                       initial_stage=initial_stage,
                                       step_before_fail=step_before_fail)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn) for insn in self.instructions)


class CodeGenerator(object):
    """Base class for code generation."""

    class CodeGenerationError(Exception):

        def __init__(self, errors):
            self.errors = errors

        def __str__(self):
            return 'Errors encountered in input to code generator.\n' + \
                '\n'.join(self.errors)

    def __init__(self, emitter, optimize=True, suppress_warnings=False):
        self.emitter = emitter
        self.suppress_warnings = suppress_warnings
        self.optimize = optimize

    def __call__(self, code):
        dag = code.instructions
        self.verify_code(code)
        extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()
        new_code = NewTimeIntegratorCode.from_old(code)

        # Generate initialization code.
        initialization_deps = code.initialization_dep_on
        initialization = extractor(dag, initialization_deps)
        initialization_code = assembler(initialization, initialization_deps)
        if self.optimize:
            initialization_code = optimizer(initialization_code)
        self.emitter.emit_initialization(initialization_code)

        # Generate timestepper code.
        stepper_deps = code.step_dep_on
        stepper = extractor(dag, code.step_dep_on)
        stepper_code = assembler(stepper, stepper_deps)
        if self.optimize:
            stepper_code = optimizer(stepper_code)
        self.emitter.emit_stepper(stepper_code)

        return self.emitter.get_code()

    def verify_code(self, code):
        """Verify that the DAG is well-formed."""
        verifier = InstructionDAGVerifier(code.instructions,
                                          code.initialization_dep_on,
                                          code.step_dep_on)
        if verifier.warnings and not self.suppress_warnings:
            for warning in verifier.warnings:
                warn(warning)
        if verifier.errors:
            raise CodeGenerator.CodeGenerationError(verifier.errors)


class StructuredCodeGenerator(object):
    """Event-driven code generation for structured languages"""

    def __call__(self, dag):
        self.verify_dag(dag)
        dag_extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()
        structural_extractor = StructuralExtractor()

        for stage, dependencies in code.stages.iteritems():
            code = extractor(dag, dependencies)
            control_flow_graph = assembler(code, dependencies)
            if self.optimize:
                control_flow_graph = optimizer(control_flow_graph)
            control_tree = structural_extractor(control_tree)
            self.lower_function(stage, control_tree)

    def lower_function(self, function_name, control_tree):
        self.emit_def_begin(function_name)
        self.emit_prolog()
        self.lower_node(control_tree)
        self.emit_epilog()
        self.emit_def_end()

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
            raise TODO('Implement lowering for unstructured intervals')

    def lower_inst(self, inst):
        if isinstance(inst, AssignInst):
            assignment = inst.assignment
            if isinstance(assignment, tuple):
                self.emit_assign_expr(assignment[0], assignment[1])
            elif isinstance(assignment, AssignExpression):
                self.emit_assign_expr(assignment.assignee,
                                      assignment.expression)
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
            self.emit_if_begin(inst.condition))
        elif isinstance(inst, ReturnInst):
            self.emit_return(inst.expression)

    # Emit routines (to be implemented by subclass)

    def start_emit(self):
        raise NotImplementedError()

    def end_emit(self):
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

    def emit_assign_rhs(self, name, rhs, time, arg):
        raise NotImplementedError()

    def emit_return(self, expr):
        raise NotImplementedError()
