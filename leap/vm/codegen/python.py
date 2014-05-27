"""Python code generator"""

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

from .expressions import PythonExpressionMapper
from .codegen_base import CodeGenerator
from .ir import AssignInst, JumpInst, BranchInst, ReturnInst, \
    UnreachableInst
from pytools.py_codegen import PythonCodeGenerator as PythonEmitter
from pytools.py_codegen import PythonFunctionGenerator as PythonFunctionEmitter
from pytools.py_codegen import Indentation
from leap.vm.utils import is_state_variable, get_unique_name
from leap.vm.language import AssignExpression, AssignRHS


class PythonCodeGenerator(CodeGenerator):
    """Converts an instruction DAG to Python code."""

    def __init__(self, method_name='Method', **kwargs):
        super(PythonCodeGenerator, self).__init__(self, **kwargs)
        import string
        self.ident_chars = set('_' + string.ascii_letters + string.digits)
        self.class_emitter = PythonClassEmitter(method_name)
        self.finished = False
        self.rhs_map = {}
        self.global_map = {}

    def name_global(self, var):
        assert is_state_variable(var)
        if var in self.global_map:
            return self.global_map[var]
        elif var == '<t>':
            self.global_map[var] = 'self.t'
        elif var == '<dt>':
            self.global_map[var] = 'self.dt'
        else:
            base = 'self.global' + self.filter_variable_name(var)
            self.global_map[var] = get_unique_name(base, self.global_map)
        return self.global_map[var]

    def filter_variable_name(self, var):
        """Converts a variable to a Python identifier."""
        return ''.join(map(lambda c: c if c in self.ident_chars else '_', var))

    def name_variables(self, symbol_table):
        """Returns a mapping from variable names to Python identifiers."""
        name_map = {}
        for var in symbol_table:
            if is_state_variable(var):
                name_map[var] = self.name_global(var)
                continue
            base = 'v_' + self.filter_variable_name(var)
            name_map[var] = get_unique_name(base, name_map)
        return name_map

    def name_rhss(self, rhss):
        """Returns a mapping from right hand side names to Python identifiers.
        """
        for rhs in rhss:
            if rhs in self.rhs_map:
                continue
            base = 'self.rhs_' + self.filter_variable_name(rhs)
            self.rhs_map[rhs] = get_unique_name(base, self.rhs_map)

    def get_globals(self, variable_set):
        """Returns the global variables in the given sequence of variable
        names."""
        return set(filter(is_state_variable, variable_set))

    def emit_function(self, var, args, control_flow_graph, name_map, rhs_map):
        """Emit the code for a function."""
        mapper = PythonExpressionMapper(name_map, numpy='self.numpy')
        emit = PythonFunctionEmitter(var, args)
        # Emit the control-flow graph as a finite state machine. The state is
        # the current basic block number. Everything is processed in a single
        # outer loop.
        emit('state = 0')
        emit('while True:')
        emit.indent()
        first = True
        for block in control_flow_graph:
            # Emit a single basic block as a sequence of instructions finished
            # by a control transfer.
            emit('%sif state == %d:' % ('el' if not first else '',
                                        block.number))
            first = False
            emit.indent()
            for inst in block:
                if isinstance(inst, JumpInst):
                    # Jumps transfer state.
                    emit('state = %i' % inst.dest.number)
                    emit('continue')

                elif isinstance(inst, BranchInst):
                    # Branches transfer state.
                    emit('state = %i if (%s) else %i' %
                         (inst.on_true.number, mapper(inst.condition),
                          inst.on_false.number))
                    emit('continue')

                elif isinstance(inst, ReturnInst):
                    emit('return %s' % mapper(inst.expression))

                elif isinstance(inst, UnreachableInst):
                    # Unreachable instructions should never be executed.
                    emit('raise RuntimeError("Entered an unreachable state!")')

                elif isinstance(inst, AssignInst):
                    self.emit_assignment(emit, inst.assignment, name_map,
                                         rhs_map, mapper)

            emit.dedent()
        emit.dedent()
        self.class_emitter.incorporate(emit)

    def emit_assignment(self, emit, assignment, name_map, rhs_map, mapper):
        """Generate the Python code for an assignment instruction."""

        if isinstance(assignment, tuple):
            var_name = name_map[assignment[0]]
            expr = mapper(assignment[1])
            emit('%s = %s' % (var_name, expr))
        elif isinstance(assignment, AssignExpression):
            var_name = name_map[assignment.assignee]
            expr = mapper(assignment.expression)
            emit('%s = %s' % (var_name, expr))
        elif isinstance(assignment, AssignRHS):
            # Get the var of the RHS and time
            rhs = rhs_map[assignment.component_id]
            time = mapper(assignment.t)

            # Build list of assignees
            assignees = map(name_map.__getitem__, assignment.assignees)

            # Build up each RHS call
            calls = []
            for argv in assignment.rhs_arguments:
                build_kwarg = lambda pair: '%s=%s' % (pair[0], mapper(pair[1]))
                if len(argv) > 0:
                    argument_string = ', '.join(map(build_kwarg, argv))
                    calls.append('%s(%s, %s)' % (rhs, time, argument_string))
                else:
                    calls.append('%s(%s)' % (rhs, time))

            # Emit the assignment
            for assignee, call in zip(assignees, calls):
                emit('%s = %s' % (assignee, call))

    def emit_constructor(self):
        emit = PythonFunctionEmitter('__init__', ('self', 'rhs_map'))
        # Perform necessary imports.
        emit('import numpy')
        emit('self.numpy = numpy')
        emit('from leap.vm.exec_numpy import StateComputed, StepCompleted')
        emit('self.StateComputed = StateComputed')
        emit('self.StepCompleted = StepCompleted')
        # Save all the rhs components.
        for rhs in self.rhs_map:
            emit('%s = rhs_map["%s"]' % (self.rhs_map[rhs], rhs))
        emit('return')
        self.class_emitter.incorporate(emit)

    def emit_set_up_function(self):
        emit = PythonFunctionEmitter('set_up', ('self', '**kwargs'))
        emit('self.t_start = kwargs["t_start"]')
        emit('self.dt_start = kwargs["dt_start"]')
        emit('self.t = self.t_start')
        emit('self.dt = self.dt_start')
        emit('state = kwargs["state"]')
        # Save all the state components.
        for state in self.global_map:
            if state == '<t>' or state == '<dt>' or state.startswith('<p>'):
                continue
            elif state.startswith('<state>'):
                emit('%s = state["%s"]' % (self.global_map[state], state[7:]))
        emit('return')
        self.class_emitter.incorporate(emit)

    def emit_run_function(self):
        emit = PythonFunctionEmitter('run', ('self', '**kwargs'))
        emit('t_end = kwargs["t_end"]')
        emit('last_step = False')
        emit('while True:')
        with Indentation(emit):
            emit('if self.t + self.dt >= t_end:')
            with Indentation(emit):
                emit('assert self.t <= t_end')
                emit('self.dt = t_end - self.t')
                emit('last_step = True')
            emit('step = self.step()')
            emit('yield self.StateComputed(t=step[0], time_id=step[1], ' +
                 'component_id=step[2], state_component=step[3])')
            emit('if last_step:')
            with Indentation(emit):
                emit('yield self.StepCompleted(t=self.t)')
                emit('break')
        self.class_emitter.incorporate(emit)

    def emit_initialization(self, control_flow_graph):
        symbol_table = control_flow_graph.symbol_table
        name_map = self.name_variables(symbol_table)
        self.name_rhss(symbol_table.rhs_names)
        self.emit_function('initialize', ('self',), control_flow_graph,
                           name_map, self.rhs_map)

    def emit_stepper(self, control_flow_graph):
        symbol_table = control_flow_graph.symbol_table
        name_map = self.name_variables(symbol_table)
        self.name_rhss(symbol_table.rhs_names)
        self.emit_function('step', ('self',), control_flow_graph, name_map,
                           self.rhs_map)

    def get_code(self):
        if not self.finished:
            self.emit_constructor()
            self.emit_set_up_function()
            self.emit_run_function()
            self.finished = True
        return self.class_emitter.get()


class PythonClassEmitter(PythonEmitter):
    """Emits code for a Python class."""

    def __init__(self, class_name, superclass='object'):
        super(PythonClassEmitter, self).__init__()
        self.class_name = class_name
        self('class %s(%s):' % (class_name, superclass))
        self.indent()

    def incorporate(self, sub_generator):
        """Add the code contained by the subgenerator while respecting the
        current level of indentation."""
        for line in sub_generator.code:
            self(line)
