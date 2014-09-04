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
from pytools.py_codegen import PythonCodeGenerator as PythonEmitter
from pytools.py_codegen import PythonFunctionGenerator as PythonFunctionEmitter
from pytools.py_codegen import Indentation
from leap.vm.utils import is_state_variable, get_unique_name
import re


def pad(line, width):
    line += ' ' * (width - 1 - len(line))
    line += '\\'
    return line


def wrap_line(line, level=0, width=80, indentation='    '):
    """The input is a line of Python code at the given indentation
    level. Return the list of lines that results from wrapping the line to the
    given width. Lines subsequent to the first line in the returned list are
    padded with extra indentation. The initial indentation level is not
    included in the input or output lines.

    Note: This code does not correctly handle Python lexical elements that
    contain whitespace.
    """
    tokens = re.split('\s+', line)
    resulting_lines = []
    at_line_start = True
    indentation_len = len(level * indentation)
    current_line = ''
    padding_width = width - indentation_len
    for index, word in enumerate(tokens):
        has_next_word = index < len(tokens) - 1
        word_len = len(word)
        if not at_line_start:
            next_len = indentation_len + len(current_line) + 1 + word_len
            if next_len < width or (not has_next_word and next_len == width):
                # The word goes on the same line.
                current_line += ' ' + word
            else:
                # The word goes on the next line.
                resulting_lines.append(pad(current_line, padding_width))
                at_line_start = True
                current_line = indentation
        if at_line_start:
            current_line += word
            at_line_start = False
    resulting_lines.append(current_line)
    return resulting_lines


class PythonClassEmitter(PythonEmitter):
    """Emits code for a Python class."""

    def __init__(self, class_name, superclass='object'):
        super(PythonClassEmitter, self).__init__()
        self.class_name = class_name
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


class PythonCodeGenerator(StructuredCodeGenerator):

    def __init__(self, class_name, optimize=True, suppress_warnings=False):
        super(PythonCodeGenerator, self).__init__(class_name, optimize,
                                                  suppress_warnings)
        # Used for emitting the method class
        self.class_emitter = PythonClassEmitter(self.class_name)
        # Map from variable / RHS names to names in generated code
        self.name_manager = PythonNameManager()
        # Expression mapper
        self.expr_mapper = PythonExpressionMapper(self.name_manager,
                                                  numpy='self.numpy')

    def expr(self, expr):
        return self.expr_mapper(expr)

    def rhs(self, rhs):
        return self.name_manager.name_rhs(rhs)

    def emit(self, line):
        level = self.class_emitter.level + self.emitter.level
        for wrapped_line in wrap_line(line, level):
            self.emitter(wrapped_line)

    def begin_emit(self):
        pass

    def emit_constructor(self):
        """Emit the constructor."""
        emit = PythonFunctionEmitter('__init__', ('self', 'rhs_map'))
        # Perform necessary imports.
        emit('import numpy')
        emit('self.numpy = numpy')
        emit('from leap.vm.exec_numpy import StateComputed, StepCompleted')
        emit('self.StateComputed = StateComputed')
        emit('self.StepCompleted = StepCompleted')
        # Save all the rhs components.
        rhs_map = self.name_manager.rhs_map
        for rhs_id, rhs in rhs_map.iteritems():
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
        for component_id, component in global_map.iteritems():
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
                emit('result = dict(result)')
                emit('t = result["time"]')
                emit('time_id = result["time_id"]')
                emit('component_id = result["component_id"]')
                emit('state_component = result["expression"]')
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

    def finish_emit(self):
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

    def emit_assign_rhs(self, name, rhs, time, arg):
        kwargs = ', '.join('{name}={expr}'.format(name=name,
            expr=self.expr(val)) for name, val in arg)
        self.emit('{name} = {rhs}(t={t}, {kwargs})'.format(
                name=self.name_manager[name], rhs=self.rhs(rhs),
                t=self.expr(time), kwargs=kwargs))

    def emit_return(self, expr):
        self.emit('return {expr}'.format(expr=self.expr(expr)))
