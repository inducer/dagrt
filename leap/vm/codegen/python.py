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
from .utils import (wrap_line_base, exec_in_new_namespace,
                    KeyToUniqueNameMap)
from pytools.py_codegen import (
        PythonCodeGenerator as PythonEmitter,
        PythonFunctionGenerator as PythonFunctionEmitter,
        Indentation)
from leap.vm.utils import is_state_variable, TODO
from functools import partial
import six


def pad_python(line, width):
    line += ' ' * (width - 1 - len(line))
    line += '\\'
    return line

wrap_line = partial(wrap_line_base, pad_func=pad_python)

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

class TimeStepUnderflow(RuntimeError):
    pass

class _function_symbol_container(object):
    pass

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
        self._local_map = KeyToUniqueNameMap(forced_prefix='local')
        self._global_map = KeyToUniqueNameMap(forced_prefix='self.global_',
                                              start={'<t>': 'self.t',
                                                     '<dt>': 'self.dt'})
        self.function_map = KeyToUniqueNameMap()

    def name_global(self, name):
        """Return the identifier for a global variable."""
        return self._global_map.get_or_make_name_for_key(name)

    def clear_locals(self):
        del self._local_map
        self._local_map = KeyToUniqueNameMap(forced_prefix='local')

    def name_local(self, local):
        """Return the identifier for a local variable."""
        return self._local_map.get_or_make_name_for_key(local)

    def name_function(self, function):
        """Return the identifier for a function."""
        return self.function_map.get_or_make_name_for_key(function)

    def get_global_ids(self):
        """Return an iterator to the recognized global variable ids."""
        return iter(self._global_map)

    def __getitem__(self, name):
        """Provide an interface to PythonExpressionMapper to look up
        the name of a local or global variable.
        """
        if is_state_variable(name):
            return self.name_global(name)
        else:
            return self.name_local(name)


class PythonCodeGenerator(StructuredCodeGenerator):

    def __init__(self, class_name, function_registry=None):
        if function_registry is None:
            from leap.vm.function_registry import base_function_registry
            function_registry = base_function_registry

        self._class_name = class_name
        self._class_emitter = PythonClassEmitter(class_name)

        # Map from variable / RHS names to names in generated code
        self._name_manager = PythonNameManager()

        self._expr_mapper = PythonExpressionMapper(
                self._name_manager, function_registry, numpy='self._numpy')

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
        for state_name, dependencies in six.iteritems(dag.states):
            code = dag_extractor(dag.instructions, dependencies)
            function = assembler(state_name, code, dependencies)
            if optimize:
                function = optimizer(function)
            control_tree = extract_structure(function)
            self.lower_function(state_name, control_tree)

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
        return namespace[self._class_name]

    def _expr(self, expr):
        return self._expr_mapper(expr)

    def _emit(self, line):
        level = self._class_emitter.level + self._emitter.level
        for wrapped_line in wrap_line(line, level):
            self._emitter(wrapped_line)

    def begin_emit(self, dag):
        self._emit_inner_classes()

    def _emit_inner_classes(self):
        """Emit the inner classes that describe objects returned by the method."""
        emit = PythonEmitter()
        for line in _inner_class_code.splitlines():
            emit(line)
        self._class_emitter.incorporate(emit)

    def _emit_constructor(self):
        """Emit the constructor."""
        emit = PythonFunctionEmitter('__init__', ('self', 'function_map'))
        # Perform necessary imports.
        emit('import numpy')
        emit('self._numpy = numpy')

        # Make function symbols available
        emit('self._functions = self._function_symbol_container()')
        for function_id in self._name_manager.function_map:
            py_function_id = self._name_manager.name_function(function_id)
            emit('self._functions.{py_function_id} = function_map["{function_id}"]'
                    .format(
                        py_function_id=py_function_id,
                        function_id=function_id))
        emit('return')
        emit("")
        self._class_emitter.incorporate(emit)

    def _emit_set_up(self):
        """Emit the set_up() method."""
        emit = PythonFunctionEmitter('set_up',
                                     ('self', 't_start', 'dt_start', 'state'))
        emit('self.t = t_start')
        emit('self.dt = dt_start')
        # Save all the state components.
        for component_id in self._name_manager.get_global_ids():
            component = self._name_manager.name_global(component_id)
            if not component_id.startswith('<state>'):
                continue
            component_id = component_id[7:]
            emit('{component} = state["{component_id}"]'.format(
                component=component, component_id=component_id))

        emit("")
        self._class_emitter.incorporate(emit)

    def _emit_run(self):
        """Emit the run() method."""
        emit = PythonFunctionEmitter('run', ('self', '**kwargs'))
        emit('t_end = kwargs["t_end"]')
        emit('last_step = False')
        # STATE_HACK: This implementation of staging support should be replaced
        # so that the states are not hard-coded.
        emit('next_states = { "initialization": "primary", ' +
             '"primary": "primary" }')
        emit('current_state = "initialization"')
        emit('while True:')
        with Indentation(emit):
            emit('if self.t + self.dt >= t_end:')
            with Indentation(emit):
                emit('assert self.t <= t_end')
                emit('self.dt = t_end - self.t')
                emit('last_step = True')
            emit('state = getattr(self, "state_" + current_state)()')
            emit('try:')
            with Indentation(emit):
                emit('while True:')
                with Indentation(emit):
                    emit('yield next(state)')

            emit('except StopIteration:')
            with Indentation(emit):
                emit('pass')
            # STATE_HACK: Ensure that the primary state has a chance to run.
            emit('if last_step and current_state == "primary":')
            with Indentation(emit):
                emit('yield self.StepCompleted(t=self.t)')
                emit('raise StopIteration()')
            emit('current_state = next_states[current_state]')
        emit("")
        self._class_emitter.incorporate(emit)

    def _emit_initialize(self):
        # This method is not used by the class, but is here for compatibility
        # with the NumpyInterpreter interface.
        emit = PythonFunctionEmitter('initialize', ('self',))
        emit('pass')

        emit("")
        self._class_emitter.incorporate(emit)

    def finish_emit(self, dag):
        self._emit_constructor()
        self._emit_set_up()
        self._emit_initialize()
        self._emit_run()

    def get_code(self):
        return self._class_emitter.get()

    def emit_def_begin(self, name):
        self._emitter = PythonFunctionEmitter('state_' + name, ('self',))
        self._name_manager.clear_locals()

    def emit_def_end(self):
        self._emit("")
        self._class_emitter.incorporate(self._emitter)
        del self._emitter

    def emit_while_loop_begin(self, expr):
        self._emit('while {expr}:'.format(expr=self._expr(expr)))
        self._emitter.indent()

    def emit_while_loop_end(self):
        self._emitter.dedent()

    def emit_if_begin(self, expr):
        self._emit('if {expr}:'.format(expr=self._expr(expr)))
        self._emitter.indent()

    def emit_if_end(self):
        self._emitter.dedent()

    def emit_else_begin(self):
        self._emit('else:')
        self._emitter.indent()

    def emit_else_end(self):
        self._emitter.dedent()

    def emit_assign_expr(self, name, expr):
        self._emit('{name} = {expr}'.format(name=self._name_manager[name],
                                           expr=self._expr(expr)))

    def emit_return(self):
        self._emit('raise StopIteration()')
        # Ensure that Python recognizes this method as a generator function by
        # adding a yield statement. Otherwise, calling methods that do not
        # yield any values may result in raising a naked StopIteration instead
        # of the creation of a generator, which does not interact well with the
        # run() implementation.
        #
        # TODO: Python 3.3+ has "yield from ()" which results in slightly less
        # awkward syntax.
        self._emit('yield')

    def emit_yield_state(self, inst):
        self._emit('yield self.StateComputed(')
        self._emit('    t=%s,' % self._expr(inst.time))
        self._emit('    time_id=%r,' % inst.time_id)
        self._emit('    component_id=%r,' % inst.component_id)
        self._emit('    state_component=%s)' % self._expr(inst.expression))

    def emit_raise(self, error_condition, error_message):
        raise TODO('Raise() for python')

    def emit_fail_step(self):
        raise TODO('FailStep() for python')
