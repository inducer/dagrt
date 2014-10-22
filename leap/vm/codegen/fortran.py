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
from pymbolic.primitives import Call, CallWithKwargs
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
        self.function_map = {}
        import string
        self.ident_chars = set('_' + string.ascii_letters + string.digits)

    def filter_name(self, var):
        result = ''.join(map(lambda c: c if c in self.ident_chars else '_', var))
        while result and result[0] == "_":
            result = result[1:]
        if not result:
            result = "leap_var"

        return result

    def name_global(self, var):
        """Return the identifier for a global variable."""
        try:
            return self.global_map[var]
        except KeyError:
            if var == '<t>':
                named_global = 'leap_t'
            elif var == '<dt>':
                named_global = 'leap_dt'
            else:
                base = self.filter_name(var)
                named_global = get_unique_name(base, self.global_map)
            self.global_map[var] = named_global
            return named_global

    def name_local(self, var):
        """Return the identifier for a local variable."""
        return get_unique_name(self.filter_name(var), self.local_map)

    def name_function(self, var):
        """Return the identifier for a function."""
        try:
            return self.function_map[var]
        except KeyError:
            base = self._filter_name(var.name)
            named_func = get_unique_name(base, self.function_map)
            self.function_map[var] = named_func
            return named_func

    def __getitem__(self, name):
        """Provide an interface to PythonExpressionMapper to look up
        the name of a local or global variable.
        """
        if is_state_variable(name):
            return 'leap_state%'+self.name_global(name)
        else:
            return self.name_local(name)

# }}}


# {{{ custom emitters

class FortranBlockEmitter(FortranEmitter):
    def __init__(self, what, code_generator=None):
        super(FortranBlockEmitter, self).__init__()
        self.what = what

        self.code_generator = code_generator

    def __enter__(self):
        if self.code_generator is not None:
            self.code_generator.emitters.append(self)
        self.indent()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dedent()
        if self.code_generator is not None:
            self.code_generator.emitters.pop()
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
    def __init__(self, parent_emitter, what, code_generator=None):
        super(FortranSubblockEmitter, self).__init__(what, code_generator)
        self.parent_emitter = parent_emitter

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(FortranSubblockEmitter, self).__exit__(
                exc_type, exc_val, exc_tb)

        self.parent_emitter.incorporate(self)


class FortranIfEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, expr, code_generator=None):
        super(FortranIfEmitter, self).__init__(
                parent_emitter, "if", code_generator)
        self("if ({expr}) then".format(expr=expr))

    def emit_else(self):
        self.dedent()
        self('else')
        self.indent()


class FortranDoEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, expr, code_generator=None):
        super(FortranDoEmitter, self).__init__(
                parent_emitter, "do", code_generator)
        self("do while ({expr})".format(expr=expr))


class FortranSubroutineEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, name, args, code_generator=None):
        super(FortranSubroutineEmitter, self).__init__(
                parent_emitter, 'subroutine', code_generator)
        self.name = name

        self('subroutine %s(%s)' % (name, ", ".join(args)))


class FortranTypeEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, type_name, code_generator=None):
        super(FortranTypeEmitter, self).__init__(
                parent_emitter, 'type', code_generator)
        self('type {type_name}'.format(type_name=type_name))

# }}}


# {{{ code generator

class _IRFunctionDescriptor(Record):
    """
    .. attribute:: name
    .. attribute:: function
    .. attribute:: control_tree
    """


class FortranType(object):
    """
    .. attribute:: base_type
    .. attribute:: dimension

        A tuple of ``'200'``, ``'-5:5'``, or some such.
        Entries may be numeric, too.

    .. attribute:: specifiers

        A tuple of things like ``'kind(8)'``
    """

    def __init__(self, base_type, dimension, specifiers):
        self.base_type = base_type
        if dimension:
            dimension = tuple(str(d) for d in dimension)
        self.dimension = dimension
        self.specifiers = specifiers


def _preprocess_preamble(text):
    if text is None:
        return []

    if not text.startswith("\n"):
        raise ValueError("expected newline as first character "
                "in preamble")

    lines = text.split("\n")
    while lines[0].strip() == "":
        lines.pop(0)
    while lines[-1].strip() == "":
        lines.pop(-1)

    if lines:
        base_indent = 0
        while lines[0][base_indent] in " \t":
            base_indent += 1

        for line in lines[1:]:
            if line[:base_indent].strip():
                raise ValueError("inconsistent indentation in preamble")

    return [line[base_indent:] for line in lines]


class FortranCodeGenerator(StructuredCodeGenerator):

    def __init__(self, module_name,
            function_registry,
            ode_component_type_map,
            module_preamble=None,
            real_scalar_kind="8",
            complex_scalar_kind="8",
            use_complex_scalars=True):
        """
        :arg rhs_id_to_component_id: a function to map
            :attr:`leap.vm.expression.RHSEvaluation.rhs_id`
            to an ODE component whose right hand side it
            computes. Identity if not given.
        :arg ode_component_type_map: a map from ODE component_id names
            to tuples, the first entry of which is the type of the
            variable, the second is the shape (as another tuple of strings)
            and further of which are type specifiers
            such as 'kind(...)'.
        """

        self.module_name = module_name
        self.function_registry = function_registry
        self.ode_component_type_map = ode_component_type_map

        self.module_preamble = _preprocess_preamble(module_preamble)

        self.real_scalar_kind = real_scalar_kind
        self.complex_scalar_kind = complex_scalar_kind
        self.use_complex_scalars = use_complex_scalars

        self.name_manager = FortranNameManager()
        self.expr_mapper = FortranExpressionMapper(
                self.name_manager)

        self.module_emitter = FortranModuleEmitter(module_name)
        self.module_emitter.__enter__()

        self.emitters = [self.module_emitter]

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

    def __call__(self, dag, optimize=True):
        from .analysis import (
                verify_code,
                collect_function_names_from_dag)
        verify_code(dag)

        func_names = collect_function_names_from_dag(dag)

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
                    _IRFunctionDescriptor(
                        name=stage_name,
                        function=function,
                        control_tree=control_tree))

        # }}}

        from leap.vm.codegen.data import SymbolKindFinder

        self.sym_kind_table = SymbolKindFinder(self.function_registry)([
            fd.function for fd in fdescrs])

        self.begin_emit(dag)
        for fdescr in fdescrs:
            self.lower_function(fdescr.name, fdescr.control_tree)
        self.finish_emit(dag)

        del self.sym_kind_table

        return self.get_code()

    def lower_function(self, function_name, control_tree):
        self.current_function = function_name

        self.emit_def_begin(function_name)
        self.lower_node(control_tree)
        self.emit_def_end()

        self.current_function = None

    def begin_emit(self, dag):
        for i, stage in enumerate(dag.stages):
            self.emit("parameter (leap_stage_{stage_name} = {i})".format(
                stage_name=stage, i=i))

        self.emit('')

        with FortranTypeEmitter(
                self.emitter,
                'leap_state_type',
                self) as emit:
            emit('integer leap_stage')
            emit('')

            for identifier, sym_kind in six.iteritems(
                    self.sym_kind_table.global_table):
                self.emit_variable_decl(
                        self.name_manager.name_global(identifier),
                        sym_kind)

        if self.module_preamble:
            for l in self.module_preamble:
                self.emit(l)
            self.emit('')

    def emit_variable_decl(self, fortran_name, sym_kind,
            is_argument=False, other_specifiers=()):
        from leap.vm.codegen.data import Boolean, Scalar, ODEComponent

        type_specifiers = other_specifiers

        if isinstance(sym_kind, Boolean):
            type_name = 'logical'

        elif isinstance(sym_kind, Scalar):
            if sym_kind.is_real_valued or not self.use_complex_scalars:
                type_name = 'real'
                type_specifiers = type_specifiers + (
                        'kind(%s)' % self.real_scalar_kind,)
            else:
                type_name = 'complex'
                type_specifiers = type_specifiers + (
                        'kind(%s)' % self.complex_scalar_kind,)

        elif isinstance(sym_kind, ODEComponent):
            comp_type = self.ode_component_type_map[sym_kind.component_id]
            type_name = comp_type.base_type

            type_specifiers = type_specifiers + comp_type.specifiers

            if not is_argument:
                type_specifiers = type_specifiers + ('allocatable',)

            if comp_type.dimension:
                if is_argument:
                    type_specifiers += (
                            "dimension(%s)"
                            % ",".join(len(comp_type.dimension)*":"),)
                else:
                    type_specifiers += (
                            "dimension(%s)" % ",".join(comp_type.dimension),)

        else:
            raise ValueError("unknown variable kind: %s" % type(sym_kind).__name__)

        if type_specifiers:
            self.emit('{type_name} {type_specifier_list} :: {id}'.format(
                type_name=type_name,
                type_specifier_list=", ".join(type_specifiers),
                id=fortran_name))
        else:
            self.emit('{type_name} {id}'.format(
                type_name=type_name,
                id=fortran_name))

    def emit_constructor(self):
        with FortranSubroutineEmitter(
                self.emitter,
                'initialize', ('self', 'rhs_map')) as emit:
            rhs_map = self.name_manager.rhs_map

            for rhs_id, rhs in rhs_map.iteritems():
                emit('{rhs} = rhs_map["{rhs_id}"]'.format(rhs=rhs, rhs_id=rhs_id))
            emit('return')

    def emit_set_up(self):
        from leap.vm.codegen.data import ODEComponent

        args = ('leap_state',) + tuple(
                self.name_manager.name_global(sym)
                for sym in self.sym_kind_table.global_table)

        with FortranSubroutineEmitter(
                self.emitter,
                'set_up', args, self):

            self.emit('leap_state_type pointer :: leap_state')
            self.emit('integer leap_ierr')
            self.emit('')

            for sym, sym_kind in six.iteritems(self.sym_kind_table.global_table):
                fortran_name = self.name_manager.name_global(sym)

                self.emit_variable_decl(
                        fortran_name,
                        sym_kind, is_argument=True,
                        other_specifiers=("optional",))
            self.emit('')

            for sym, sym_kind in six.iteritems(self.sym_kind_table.global_table):
                tgt_fortran_name = self.name_manager[sym]
                fortran_name = self.name_manager.name_global(sym)

                with FortranIfEmitter(
                        self.emitter, 'present(%s)' % fortran_name, self):
                    if isinstance(sym_kind, ODEComponent):
                        self.emit_allocation_check(
                                tgt_fortran_name, sym_kind.component_id)

                    self.emit("{name} = {arg}"
                            .format(
                                name=tgt_fortran_name,
                                arg=fortran_name))

    def emit_run_step(self):
        """Emit the run() method."""
        with FortranSubroutineEmitter(
                self.emitter,
                'run', ('leap_state', 't_end'), self) as emit:
            emit('t_end = kwargs["t_end"]')
            emit('last_step = False')


            # STAGE_HACK: This implementation of staging support should be replaced
            # so that the stages are not hard-coded.
            emit('next_stages = { "initialization": "primary", ' +
                 '"primary": "primary" }')
            emit('current_stage = "initialization"')

            with FortranDoEmitter(emit, ".true.", self):
                with FortranIfEmitter(
                        self.emitter, 'self.t + self.dt >= t_end', self):
                    self.emit('assert self.t <= t_end')
                    self.emit('self.dt = t_end - self.t')
                    self.emit('last_step = True')

                d_emit('stage_function = getattr(self, "stage_" + current_stage)')
                emit('result = stage_function()')

                with FortranIfEmitter(self.emitter, 'result') as emit_res:
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
        #self.emit_constructor()
        self.emit_set_up()
        #self.emit_initialize()
        #self.emit_run()

        self.module_emitter.__exit__(None, None, None)

    def get_code(self):
        return self.module_emitter.get()

    # {{{ called by superclass

    def emit_def_begin(self, func_id):
        # The current function is handled by self.emit
        FortranSubroutineEmitter(
                self.emitter,
                'leap_stage_func_' + func_id, ('leap_state',),
                self).__enter__()

        self.emit('leap_state_type pointer :: leap_state')
        self.emit('integer leap_ierr')
        self.emit('')

        count = 0
        for identifier, sym_kind in \
                six.iteritems(
                        self.sym_kind_table.per_function_table.get(func_id, {})):
            self.emit_variable_decl(
                    self.name_manager.name_local(identifier),
                    sym_kind)
            count += 1

        if count:
            self.emit('')

    def emit_def_end(self):
        self.emitter.__exit__(None, None, None)

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

    def emit_allocation_check(self, fortran_name, component_id):
        with FortranIfEmitter(
                self.emitter, '.not.allocated(%s)' % fortran_name, self):
            comp_type = self.ode_component_type_map[component_id]

            if comp_type.dimension:
                dimension = '(%s)' % ', '.join(
                        str(dim_axis) for dim_axis in comp_type.dimension)
            else:
                dimension = ""

            self.emit('allocate({name}{dimension}, stat=leap_ierr)'.format(
                name=fortran_name,
                dimension=dimension))
            with FortranIfEmitter(self.emitter, 'leap_ierr.ne.0', self):
                self.emit("write(*,*) 'failed to allocate {name}'".format(
                    name=fortran_name))

    def emit_assign_expr(self, name, expr):
        from leap.vm.codegen.data import ODEComponent

        if expr is not None:
            fortran_name = self.name_manager[name]

            if name == 'retval':
                # FIXME
                return

            sym_kind = self.sym_kind_table.get(
                    self.current_function, name)

            if isinstance(sym_kind, ODEComponent):
                self.emit_allocation_check(fortran_name, sym_kind.component_id)

            self.emit(
                    "{name} = {expr}"
                    .format(
                        name=fortran_name,
                        expr=self.expr(expr)))
        else:
            self.emit(
                    "! unimplemented: {name} = None"
                    .format(
                        name=self.name_manager[name]))

        self.emit('')

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
