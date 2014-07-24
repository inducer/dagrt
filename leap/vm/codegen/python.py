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

import string
from .expressions import PythonExpressionMapper
from .codegen_base import CodeGenerator
from .ir import AssignInst, JumpInst, BranchInst, ReturnInst, \
    UnreachableInst
from pytools.py_codegen import PythonCodeGenerator as PythonEmitter
from pytools.py_codegen import PythonFunctionGenerator as PythonFunctionEmitter
from pytools.py_codegen import Indentation
from leap.vm.utils import is_state_variable, get_unique_name
from leap.vm.language import AssignExpression, AssignRHS


class PythonClassEmitter(PythonEmitter):
    """Emits code for a Python class."""

    def __init__(self, class_name, superclass='object'):
        super(PythonClassEmitter, self).__init__()
        self.class_name = class_name
        self('class %s(%s):' % (class_name, superclass))
        self.indent()

    def incorporate(self, sub_generator):
        """Add the code contained by the subgenerator while respecting the
        current level of indentation.
        """
        for line in sub_generator.code:
            self(line)


class PythonNameManager(object):
    """Maps names that appear in intermediate code to Python
    identifiers.
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

    def __init__(self, **kwargs):
        super(PythonCodeGenerator, self).__init__(self, **kwargs)
        # Used for emitting the method class
        self.class_emitter = PythonClassEmitter(self.method_name)
        # Map from variable / RHS names to names in generated code
        self.name_manager = PythonNameManager()
        # Expression mapper
        self.expr_mapper = PythonExpressionMapper(self.name_manager,
                                                  numpy='self.numpy')

    def expr(self, expr):
        return self.expr_mapper(expr)

    def rhs(self, rhs):
        return self.name_manager.name_rhs(rhs)

    def begin_emit(self):
        pass

    def emit_StateComputed(self):
        pass

    def emit_StepCompleted(self):
        pass

    def emit_constructor(self):
        """Emit the constructor."""
        emit = PythonFunctionEmitter('__init__', ('self', 'rhs_map'))
        # Perform necessary imports.
        emit('import numpy')
        emit('self.numpy = numpy')
        # XXX this should be an inner class
        emit('from leap.vm.exec_numpy import StateComputed, StepCompleted')
        emit('self.StateComputed = StateComputed')
        emit('self.StepCompleted = StepCompleted')
        # Save all the rhs components.
        for rhs in self.rhs_map:
            emit('%s = rhs_map["%s"]' % (self.rhs_map[rhs], rhs))
        emit('return')
        self.class_emitter.incorporate(emit)

    def emit_set_up(self):
        """Emit the set_up() method."""

    def emit_run(self):
        """Emit the run() method."""
        # XXX respect stages.
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

    def finish_emit(self):
        self.emit_constructor()
        self.emit_set_up()
        self.emit_run()

    def emit_def_begin(self, name):
        # The current function is handled by self.emit
        self.emit = PythonFunctionEmitter(name, ('self',))

    def emit_def_end(self):
        self.class_emitter.incorporate(self.emit)
        del self.emit

    def emit_while_loop_begin(self, expr):
        self.emit('while {expr}:'.format(expr=self.expr(expr)))
        self.emit.indent()
    
    def emit_while_loop_end(self):
        self.emit.dedent()

    def emit_if_begin(self, expr):
        self.emit('if {expr}:'.format(expr=self.expr(expr)))
        self.emit.indent()
    
    def emit_if_end(self):
        self.emit.dedent()

    def emit_else_begin(self):
        self.emit('else:')
        self.emit.indent()

    def emit_else_end(self):
        self.emit.dedent()

    def emit_assign_expr(self, name, expr):
        self.emit('{name} = {expr}'.format(name=self.expr(name),
                                           expr=self.expr(expr)))

    def emit_assign_rhs(self, name, rhs, time, arg):
        self.emit('{name} = {rhs}(t={t}, {expr})'.format(name=self.expr(name),
                                                         rhs=self.rhs(rhs),
                                                         t=self.expr(time),
                                                         expr=self.expr(arg))

    def emit_return(self, expr):
        self.emit('return {expr}'.format(expr=self.expr(expr)))
