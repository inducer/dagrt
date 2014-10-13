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
from .codegen_base import StructuredCodeGenerator
from pytools.py_codegen import (
        PythonCodeGenerator as PythonEmitter,
        PythonFunctionGenerator as PythonFunctionEmitter,
        Indentation)
from leap.vm.utils import is_state_variable, get_unique_name
from leap.vm.codegen.utils import wrap_line_base
from functools import partial
import six


def pad_python(line, width):
    line += ' ' * (width - 1 - len(line))
    line += '\\'
    return line

wrap_line = partial(wrap_line_base, pad_python)

_inner_class_code = '''from collections import namedtuple

class StateComputed(namedtuple("StateComputed",
         ["t", "time_id", "component_id", "state_component"])):
    """
    .. attribute:: t
    .. attribute:: time_id
    .. attribute:: component_id

        Identifier of the state component being returned.

    .. attribute:: state_component
    """

class StepCompleted(namedtuple("StepCompleted", ["t"])):
    """
    .. attribute:: t

        Floating point number.
    """

class StepFailed(namedtuple("StepFailed", ["t"])):
    """
    .. attribute:: t

        Floating point number.
    """
'''


class PythonClassEmitter(PythonEmitter):
    """Emits code for a Python class."""

    def __init__(self, class_name, superclass='object'):
        super(PythonClassEmitter, self).__init__()
        self('class {cls}({superclass}):'.format(cls=class_name,
                                                 superclass=superclass))
        self.indent()

    def incorporate(self, sub_generator):
        """Add the code contained by the subgenerator while respecting the
        current level of indentation.
        """
        for line in sub_generator.code:
            self(line)


class PythonNameManager(object):
    """Maps names that appear in intermediate code to Python identifiers.
    """

    def __init__(self):
        self.local_map = {}
        self.global_map = {}
        self.rhs_map = {}
        import string
        self.ident_chars = set('_' + string.ascii_letters + string.digits)

    def filter_name(self, var):
        return ''.join(map(lambda c: c if c in self.ident_chars else '', var))

    def name_global(self, var):
        """Return the identifier for a global variable."""
        try:
            return self.global_map[var]
        except KeyError:
            if var == '<t>':
                named_global = 'self.t'
            elif var == '<dt>':
                named_global = 'self.dt'
            else:
                base = 'self.global_' + self.filter_name(var)
                named_global = get_unique_name(base, self.global_map)
            self.global_map[var] = named_global
            return named_global

    def clear_locals(self):
        self.local_map.clear()

    def name_local(self, var):
        """Return the identifier for a local variable."""
        base = 'local_' + self.filter_name(var)
        return get_unique_name(base, self.local_map)

    def name_rhs(self, var):
        """Return the identifier for an RHS."""
        try:
            return self.rhs_map[var]
        except KeyError:
            base = 'self.rhs_' + self.filter_name(var)
            named_rhs = get_unique_name(base, self.rhs_map)
            self.rhs_map[var] = named_rhs
            return named_rhs

    def __getitem__(self, name):
        """Provide an interface to PythonExpressionMapper to look up
        the name of a local or global variable.
        """
        if is_state_variable(name):
            return self.name_global(name)
        else:
            return self.name_local(name)


def exec_in_new_namespace(code):
    namespace = {}
    exec(code, globals(), namespace)
    return namespace


class PythonCodeGenerator(StructuredCodeGenerator):

    def __init__(self, class_name):
        self.class_name = class_name

        self.class_emitter = PythonClassEmitter(class_name)

        # Map from variable / RHS names to names in generated code
        self.name_manager = PythonNameManager()

        self.expr_mapper = PythonExpressionMapper(self.name_manager,
                                                  numpy='self.numpy')

    def __call__(self, dag, optimize=True):
        from .analysis import verify_code
        verify_code(dag)

        from .codegen_base import NewTimeIntegratorCode
        dag = NewTimeIntegratorCode.from_old(dag)

        from .dag2ir import InstructionDAGExtractor, ControlFlowGraphAssembler
        from .optimization import Optimizer
        from .ir2structured_ir import StructuralExtractor

        dag_extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()
        extract_structure = StructuralExtractor()

        self.begin_emit(dag)
        for stage_name, dependencies in six.iteritems(dag.stages):
            code = dag_extractor(dag.instructions, dependencies)
            function = assembler(stage_name, code, dependencies)
            if optimize:
                function = optimizer(function)
            control_tree = extract_structure(function)
            self.lower_function(stage_name, control_tree)

        self.finish_emit(dag)

        return self.get_code()

    def lower_function(self, function_name, control_tree):
        self.emit_def_begin(function_name)
        self.lower_node(control_tree)
        self.emit_def_end()

    def get_class(self, code):
        """Return the compiled Python class for the method."""
        python_code = self(code)
        namespace = exec_in_new_namespace(python_code)
        return namespace[self.class_name]

    def expr(self, expr):
        return self.expr_mapper(expr)

    def rhs(self, rhs):
        return self.name_manager.name_rhs(rhs)

    def emit(self, line):
        level = self.class_emitter.level + self.emitter.level
        for wrapped_line in wrap_line(line, level):
            self.emitter(wrapped_line)

    def begin_emit(self, dag):
        self.emit_inner_classes()

    def emit_inner_classes(self):
        """Emit the inner classes that describe objects returned by the method."""
        emit = PythonEmitter()
        for line in _inner_class_code.splitlines():
            emit(line)
        self.class_emitter.incorporate(emit)

    def emit_constructor(self):
        """Emit the constructor."""
        emit = PythonFunctionEmitter('__init__', ('self', 'rhs_map'))
        # Perform necessary imports.
        emit('import numpy')
        emit('self.numpy = numpy')
        # Save all the rhs components.
        rhs_map = self.name_manager.rhs_map
        for rhs_id, rhs in six.iteritems(rhs_map):
            emit('{rhs} = rhs_map["{rhs_id}"]'.format(rhs=rhs, rhs_id=rhs_id))
        emit('return')
        self.class_emitter.incorporate(emit)

    def emit_set_up(self):
        """Emit the set_up() method."""
        emit = PythonFunctionEmitter('set_up',
                                     ('self', 't_start', 'dt_start', 'state'))
        emit('self.t = t_start')
        emit('self.dt = dt_start')
        # Save all the state components.
        global_map = self.name_manager.global_map
        for component_id, component in six.iteritems(global_map):
            if not component_id.startswith('<state>'):
                continue
            component_id = component_id[7:]
            emit('{component} = state["{component_id}"]'.format(
                component=component, component_id=component_id))
        self.class_emitter.incorporate(emit)

    def emit_run(self):
        """Emit the run() method."""
        emit = PythonFunctionEmitter('run', ('self', '**kwargs'))
        emit('t_end = kwargs["t_end"]')
        emit('last_step = False')
        # STAGE_HACK: This implementation of staging support should be replaced
        # so that the stages are not hard-coded.
        emit('next_stages = { "initialization": "primary", ' +
             '"primary": "primary" }')
        emit('current_stage = "initialization"')
        emit('while True:')
        with Indentation(emit):
            emit('if self.t + self.dt >= t_end:')
            with Indentation(emit):
                emit('assert self.t <= t_end')
                emit('self.dt = t_end - self.t')
                emit('last_step = True')
            emit('stage_function = getattr(self, "stage_" + current_stage)')
            emit('result = stage_function()')
            emit('if result:')
            with Indentation(emit):
                emit('t = result[0][1]')
                emit('time_id = result[1][1]')
                emit('component_id = result[2][1]')
                emit('state_component = result[3][1]')
                emit('yield self.StateComputed(t=t, time_id=time_id, \\')
                emit('    component_id=component_id, ' +
                     'state_component=state_component)')
                emit('if last_step:')
                with Indentation(emit):
                    emit('yield self.StepCompleted(t=self.t)')
                    emit('break')
            emit('current_stage = next_stages[current_stage]')
        self.class_emitter.incorporate(emit)

    def emit_initialize(self):
        # This method is not used by the class, but is here for compatibility
        # with the NumpyInterpreter interface.
        emit = PythonFunctionEmitter('initialize', ('self',))
        emit('pass')
        self.class_emitter.incorporate(emit)

    def finish_emit(self, dag):
        self.emit_constructor()
        self.emit_set_up()
        self.emit_initialize()
        self.emit_run()

    def get_code(self):
        return self.class_emitter.get()

    def emit_def_begin(self, name):
        # The current function is handled by self.emit
        self.emitter = PythonFunctionEmitter('stage_' + name, ('self',))

    def emit_def_end(self):
        self.class_emitter.incorporate(self.emitter)
        del self.emitter

    def emit_while_loop_begin(self, expr):
        self.emit('while {expr}:'.format(expr=self.expr(expr)))
        self.emitter.indent()

    def emit_while_loop_end(self):
        self.emitter.dedent()

    def emit_if_begin(self, expr):
        self.emit('if {expr}:'.format(expr=self.expr(expr)))
        self.emitter.indent()

    def emit_if_end(self):
        self.emitter.dedent()

    def emit_else_begin(self):
        self.emit('else:')
        self.emitter.indent()

    def emit_else_end(self):
        self.emitter.dedent()

    def emit_assign_expr(self, name, expr):
        self.emit('{name} = {expr}'.format(name=self.name_manager[name],
                                           expr=self.expr(expr)))

    def emit_assign_norm(self, name, expr, p):
        # NOTE: Doesn't handle inf.
        self.emit('{name} = self.numpy.linalg.norm({expr}, ord={ord})'.format(
                name=self.name_manager[name], expr=self.expr(expr), ord=p))

    def emit_assign_rhs(self, name, rhs, time, arg):
        kwargs = ', '.join('{name}={expr}'.format(name=name,
            expr=self.expr(val)) for name, val in arg)
        self.emit('{name} = {rhs}(t={t}, {kwargs})'.format(
                name=self.name_manager[name], rhs=self.rhs(rhs),
                t=self.expr(time), kwargs=kwargs))

    def emit_return(self, expr):
        self.emit('return {expr}'.format(expr=self.expr(expr)))
