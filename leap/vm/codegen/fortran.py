"""Fortran code generator"""

__copyright__ = "Copyright (C) 2014 Matt Wala, Andreas Kloeckner"

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

from .expressions import FortranExpressionMapper
from .codegen_base import StructuredCodeGenerator
from leap.vm.utils import is_state_variable, get_unique_name
from pytools.py_codegen import (
        # It's the same code. So sue me.
        PythonCodeGenerator as FortranEmitter)
from .utils import wrap_line_base
from functools import partial
import re  # noqa
import six

from pytools import Record


def pad_fortran(line, width):
    line += ' ' * (width - 1 - len(line))
    line += '&'
    return line

wrap_line = partial(wrap_line_base, pad_fortran)


# {{{ name manager

class FortranNameManager(object):
    """Maps names that appear in intermediate code to Fortran identifiers.
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

# }}}


# {{{ custom emitters

class FortranBlockEmitter(FortranEmitter):
    def __init__(self, what):
        super(FortranBlockEmitter, self).__init__()
        self.what = what

    def __enter__(self):
        self.indent()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dedent()
        self('end {what}'.format(what=self.what))
        self('')

    def incorporate(self, sub_generator):
        for line in sub_generator.code:
            self(line)


class FortranModuleEmitter(FortranBlockEmitter):
    def __init__(self, module_name):
        super(FortranModuleEmitter, self).__init__('module')
        self.module_name = module_name
        self('module {module_name}'.format(module_name=module_name))


class FortranSubblockEmitter(FortranBlockEmitter):
    def __init__(self, parent_emitter, what):
        super(FortranSubblockEmitter, self).__init__(what)
        self.parent_emitter = parent_emitter

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(FortranSubblockEmitter, self).__exit__(
                exc_type, exc_val, exc_tb)

        self.parent_emitter.incorporate(self)


class FortranIfEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, expr):
        super(FortranIfEmitter, self).__init__(parent_emitter, "if")
        self("if ({expr}) then".format(expr=expr))

    def emit_else(self):
        self.dedent()
        self('else')
        self.indent()


class FortranDoEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, expr):
        super(FortranDoEmitter, self).__init__(parent_emitter, "do")
        self("do while ({expr})".format(expr=expr))


class FortranSubroutineEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, name, args):
        super(FortranSubroutineEmitter, self).__init__(
                parent_emitter, 'subroutine')
        self.name = name

        self('subroutine %s(%s)' % (name, ", ".join(args)))


class FortranTypeEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, type_name):
        super(FortranTypeEmitter, self).__init__(parent_emitter, 'type')
        self('type {type_name}'.format(type_name=type_name))

# }}}


# {{{ code generator

class _FunctionDescriptor(Record):
    """
    .. attribute:: name
    .. attribute:: function
    .. attribute:: control_tree
    """


class FortranCodeGenerator(StructuredCodeGenerator):

    def __init__(self, module_name, time_kind="8",
            rhs_id_to_component_id=None):
        """
        :arg rhs_id_to_component_id: a function to map
            :attr:`leap.vm.expression.RHSEvaluation.rhs_id`
            to an ODE component whose right hand side it
            computes. Identity if not given.
        """

        self.module_name = module_name
        self.time_kind = time_kind

        self.name_manager = FortranNameManager()
        self.expr_mapper = FortranExpressionMapper(self.name_manager)

        self.module_emitter = FortranModuleEmitter(module_name)
        self.module_emitter.__enter__()

        self.emitters = [self.module_emitter]

        if rhs_id_to_component_id is None:
            rhs_id_to_component_id = lambda name: name

        self.rhs_id_to_component_id = rhs_id_to_component_id

    @property
    def emitter(self):
        return self.emitters[-1]

    def expr(self, expr):
        return self.expr_mapper(expr)

    def rhs(self, rhs):
        return self.name_manager.name_rhs(rhs)

    def emit(self, line):
        level = sum(em.level for em in self.emitters)
        for wrapped_line in wrap_line(line, level):
            self.emitter(wrapped_line)

    def __call__(self, dag, rhs_id_to_component_id, optimize=True):
        from .analysis import (
                verify_code,
                collect_rhs_names_from_dag)
        verify_code(dag)

        rhs_names = collect_rhs_names_from_dag(dag)

        from .codegen_base import NewTimeIntegratorCode
        dag = NewTimeIntegratorCode.from_old(dag)

        # {{{ produce function descriptors

        from .dag2ir import InstructionDAGExtractor, ControlFlowGraphAssembler
        from .optimization import Optimizer
        from .ir2structured_ir import StructuralExtractor

        fdescrs = []

        dag_extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()
        extract_structure = StructuralExtractor()

        for stage_name, dependencies in six.iteritems(dag.stages):
            code = dag_extractor(dag.instructions, dependencies)
            function = assembler(stage_name, code, dependencies)
            if optimize:
                function = optimizer(function)
            control_tree = extract_structure(function)

            fdescrs.append(
                    _FunctionDescriptor(
                        name=stage_name,
                        function=function,
                        control_tree=control_tree))

        # }}}

        from leap.vm.codegen.data import SymbolKindFinder

        sym_kind_table = SymbolKindFinder(self.rhs_id_to_component_id)([
            fd.function for fd in fdescrs])

        self.begin_emit(dag)
        for fdescr in fdescrs:
            code = dag_extractor(dag.instructions, dependencies)
            function = assembler(stage_name, code, dependencies)
            if self.optimize:
                function = optimizer(function)
            control_tree = extract_structure(function)
            self.lower_function(stage_name, control_tree)

        self.finish_emit(dag)

        return self.get_code()

    def lower_function(self, function_name, control_tree):
        self.emit_def_begin(function_name)
        self.lower_node(control_tree)
        self.emit_def_end()

    def begin_emit(self, dag):
        for i, stage in enumerate(dag.stages):
            self.emit("parameter (stage_{stage_name} = {i})".format(
                stage_name=stage, i=i))

        self.emit('')

        with FortranTypeEmitter(
                self.emitter,
                'stepper_state') as emit:
            emit("integer stage")
            emit("real kind({time_kind}) t".format(time_kind=self.time_kind))
            emit("real kind({time_kind}) dt".format(time_kind=self.time_kind))

    def emit_constructor(self):
        with FortranSubroutineEmitter(
                self.emitter,
                'initialize', ('self', 'rhs_map')) as emit:
            rhs_map = self.name_manager.rhs_map

            for rhs_id, rhs in rhs_map.iteritems():
                emit('{rhs} = rhs_map["{rhs_id}"]'.format(rhs=rhs, rhs_id=rhs_id))
            emit('return')

    def emit_set_up(self):
        with FortranSubroutineEmitter(
                self.emitter,
                'set_up',
                ('self', 't_start', 'dt_start', 'state')) as emit:
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

    def emit_run(self):
        """Emit the run() method."""
        with FortranSubroutineEmitter(
                self.emitter,
                'run', ('self', '**kwargs')) as emit:
            emit('t_end = kwargs["t_end"]')
            emit('last_step = False')
            # STAGE_HACK: This implementation of staging support should be replaced
            # so that the stages are not hard-coded.
            emit('next_stages = { "initialization": "primary", ' +
                 '"primary": "primary" }')
            emit('current_stage = "initialization"')

            with FortranDoEmitter(emit, ".true.") as d_emit:
                with FortranIfEmitter(
                        d_emit, 'self.t + self.dt >= t_end') as emit_over:
                    emit_over('assert self.t <= t_end')
                    emit_over('self.dt = t_end - self.t')
                    emit_over('last_step = True')

                d_emit('stage_function = getattr(self, "stage_" + current_stage)')
                emit('result = stage_function()')

                with FortranIfEmitter(d_emit, 'result') as emit_res:
                    emit_res('result = dict(result)')
                    emit_res('t = result["time"]')
                    emit_res('time_id = result["time_id"]')
                    emit_res('component_id = result["component_id"]')
                    emit_res('state_component = result["expression"]')
                    emit_res('yield self.StateComputed(t=t, time_id=time_id, \\')
                    emit_res('    component_id=component_id, ' +
                         'state_component=state_component)')

                    with FortranIfEmitter(emit_res, 'last_step') as emit_ls:
                        emit_ls('yield self.StepCompleted(t=self.t)')
                        emit_ls('exit')
                d_emit('current_stage = next_stages[current_stage]')

    def emit_initialize(self):
        pass

    def finish_emit(self, dag):
        self.emit_constructor()
        self.emit_set_up()
        self.emit_initialize()
        self.emit_run()

        self.module_emitter.__exit__(None, None, None)

    def get_code(self):
        return self.module_emitter.get()

    # {{{ called by superclass

    def emit_def_begin(self, name):
        # The current function is handled by self.emit
        self.emitters.append(FortranSubroutineEmitter(
                self.emitter,
                'stage_' + name, ('self',)))
        self.emitter.__enter__()

    def emit_def_end(self):
        self.emitter.__exit__(None, None, None)
        self.emitters.pop()

    def emit_while_loop_begin(self, expr):
        self.emitter = FortranDoEmitter(
                self.emitter,
                self.expr(expr))
        self.emitter.__enter__()

    emit_while_loop_end = emit_def_end

    def emit_if_begin(self, expr):
        self.emitter = FortranIfEmitter(
                self.emitter,
                self.expr(expr))

    emit_if_end = emit_def_end

    def emit_else_begin(self):
        self.emitter.emit_else()

    emit_else_end = emit_def_end

    def emit_assign_expr(self, name, expr):
        if expr is not None:
            self.emit(
                    "{name} = {expr}"
                    .format(
                        name=self.name_manager[name],
                        expr=self.expr(expr)))
        else:
            self.emit(
                    "! unimplemented: {name} = None"
                    .format(
                        name=self.name_manager[name]))

    def emit_assign_rhs(self, name, rhs, time, arg):
        kwargs = ', '.join('{name}={expr}'.format(name=name,
            expr=self.expr(val)) for name, val in arg)
        self.emit('{name} = {rhs}(t={t}, {kwargs})'.format(
                name=self.name_manager[name], rhs=self.rhs(rhs),
                t=self.expr(time), kwargs=kwargs))

    def emit_return(self, expr):
        self.emit('return {expr}'.format(expr=self.expr(expr)))

    # }}}

# }}}

# vim: foldmethod=marker
