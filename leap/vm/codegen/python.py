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
from .ir import YieldStateInst
from pytools.py_codegen import (
        PythonCodeGenerator as PythonEmitter,
        PythonFunctionGenerator as PythonFunctionEmitter,
        Indentation)
from leap.vm.utils import is_state_variable
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

class StepCompleted(
        namedtuple("StepCompleted",
            ["t", "current_state", "next_state"])):
    """
    .. attribute:: t

        Approximate integrator time at end of step.

    .. attribute:: current_state
    .. attribute:: next_state
    """

class StepFailed(namedtuple("StepFailed", ["t"])):
    """
    .. attribute:: t

        Floating point number.
    """


class TimeStepUnderflow(RuntimeError):
    pass


class FailStepException(RuntimeError):
    pass


class TransitionEvent(Exception):

    def __init__(self, next_state):
        self.next_state = next_state


class _function_symbol_container(object):
    pass

def _builtin_norm(self, x, ord=None):
    if self._numpy.isscalar(x):
        return abs(x)
    return self._numpy.linalg.norm(x, ord)
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
        self.function_map = KeyToUniqueNameMap(forced_prefix="self._functions.")

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


class BareExpression(object):
    """The point of this class is to have a ``repr()`` that is exactly
    a given string. That way, ``repr()`` can be used as a helper for
    code generation on structured data.
    """

    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


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

        from .dag2ir import InstructionDAGExtractor, ControlFlowGraphAssembler
        from .optimization import Optimizer
        from .ir2structured_ir import StructuralExtractor

        dag_extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()
        extract_structure = StructuralExtractor()

        self.begin_emit(dag)
        for state_name, state in six.iteritems(dag.states):
            code = dag_extractor(dag.instructions, state.depends_on)
            function = assembler(state_name, code, state.depends_on)
            if optimize:
                function = optimizer(function)
            self._pre_lower(function)
            control_tree = extract_structure(function)
            self.lower_function(state_name, control_tree)

        self.finish_emit(dag)

        return self.get_code()

    def _pre_lower(self, function):
        self._has_yield_inst = False
        for block in function:
            for inst in block:
                if isinstance(inst, YieldStateInst):
                    self._has_yield_inst = True
                    return

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

    def _emit_constructor(self, dag):
        """Emit the constructor."""
        emit = PythonFunctionEmitter('__init__', ('self', 'function_map'))
        # Perform necessary imports.
        emit('import numpy')
        emit('self._numpy = numpy')

        # Make function symbols available
        emit('self._functions = self._function_symbol_container()')
        for function_id in self._name_manager.function_map:
            py_function_id = self._name_manager.name_function(function_id)
            emit('{py_function_id} = function_map["{function_id}"]'
                    .format(
                        py_function_id=py_function_id,
                        function_id=function_id))
        emit("")
        emit("self.state_transition_table = "+repr(dict(
            (state_name, (
                state.next_state,
                BareExpression("self.state_"+state_name)))
            for state_name, state in six.iteritems(dag.states))))
        emit("")

        self._class_emitter.incorporate(emit)

    def _emit_set_up(self, dag):
        """Emit the set_up() method."""
        emit = PythonFunctionEmitter('set_up',
                                     ('self', 't_start', 'dt_start', 'context'))
        emit('self.t = t_start')
        emit('self.dt = dt_start')
        # Save all the context components.
        for component_id in self._name_manager.get_global_ids():
            component = self._name_manager.name_global(component_id)
            if not component_id.startswith('<state>'):
                continue
            component_id = component_id[7:]
            emit('{component} = context["{component_id}"]'.format(
                component=component, component_id=component_id))

        emit("self.next_state = "+repr(dag.initial_state))

        emit("")
        self._class_emitter.incorporate(emit)

    def _emit_run(self):
        emit = PythonFunctionEmitter('run', ('self', 't_end=None', 'max_steps=None'))
        emit('n_steps = 0')
        emit('while True:')
        with Indentation(emit):
            emit('if t_end is not None and self.t >= t_end:')
            with Indentation(emit):
                emit('return')

            emit('if max_steps is not None and n_steps >= max_steps:')
            with Indentation(emit):
                emit('return')

            emit('cur_state = self.next_state')
            emit('try:')
            with Indentation(emit):
                emit('for evt in self.run_single_step():')
                with Indentation(emit):
                    emit('yield evt')

            emit('except self.FailStepException:')
            with Indentation(emit):
                emit('yield self.StepFailed(t=self.t)')
                emit('continue')

            emit('except self.TransitionEvent as evt:')
            with Indentation(emit):
                emit('self.next_state = evt.next_state')

            emit('yield self.StepCompleted(t=self.t, '
                'current_state=cur_state, next_state=self.next_state)')

            emit('n_steps += 1')

        emit("")
        self._class_emitter.incorporate(emit)

    def _emit_run_single_step(self):
        emit = PythonFunctionEmitter('run_single_step', ('self',))

        emit('self.next_state, state_func = '
             'self.state_transition_table[self.next_state]')

        emit('for evt in state_func():')
        with Indentation(emit):
            emit('yield evt')
        emit("")
        self._class_emitter.incorporate(emit)

    def finish_emit(self, dag):
        self._emit_constructor(dag)
        self._emit_set_up(dag)
        self._emit_run()
        self._emit_run_single_step()

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
        self._emitter.dedent()
        self._emit('else:')
        self._emitter.indent()

    def emit_assign_expr(self, name, expr):
        self._emit('{name} = {expr}'.format(name=self._name_manager[name],
                                           expr=self._expr(expr)))

    def emit_return(self):
        self._emit('return')
        # Ensure that Python recognizes this method as a generator function by
        # adding a yield statement. Otherwise, calling methods that do not
        # yield any values may result in raising a naked StopIteration instead
        # of the creation of a generator, which does not interact well with the
        # run() implementation.
        #
        # TODO: Python 3.3+ has "yield from ()" which results in slightly less
        # awkward syntax.
        if not self._has_yield_inst:
            self._emit('yield')

    def emit_yield_state(self, inst):
        self._emit('yield self.StateComputed(t={t}, time_id={time_id}, '
                   'component_id={component_id}, '
                   'state_component={state_component})'.format(
                       t=self._expr(inst.time),
                       time_id=repr(inst.time_id),
                       component_id=repr(inst.component_id),
                       state_component=self._expr(inst.expression)))

    def emit_raise(self, error_condition, error_message):
        self._emit('raise self.{condition}("{message}")'.format(
                   condition=error_condition.__name__,
                   message=error_message))
        if not self._has_yield_inst:
            self._emit('yield')

    def emit_fail_step(self):
        self._emit('raise self.FailStepException()')
        if not self._has_yield_inst:
            self._emit('yield')

    def emit_state_transition(self, next_state):
        assert '\'' not in next_state
        self._emit('raise self.TransitionEvent(\'' + next_state + '\')')
        if not self._has_yield_inst:
            self._emit('yield')
