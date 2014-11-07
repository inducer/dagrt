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
        PythonCodeGenerator as FortranEmitterBase)
from pymbolic.primitives import Call, CallWithKwargs, Variable
from .utils import wrap_line_base
from functools import partial
import re  # noqa
import six
import string

from pytools import Record


def pad_fortran(line, width):
    line += ' ' * (width - 1 - len(line))
    line += '&'
    return line

wrap_line = partial(wrap_line_base, pad_func=pad_fortran)


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

    def name_refcount(self, name):
        """Provide an interface to PythonExpressionMapper to look up
        the name of a local or global variable.
        """
        if is_state_variable(name):
            return 'leap_state%leap_refcnt_'+self.name_global(name)
        else:
            return "leap_refcnt_"+self.name_local(name)


# }}}


# {{{ custom emitters

class FortranEmitter(FortranEmitterBase):
    def incorporate(self, sub_generator):
        for line in sub_generator.code:
            self(line)


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


class FortranCallCode(object):
    def __init__(self, template):
        from mako.template import Template

        self.template = Template(template, strict_undefined=True)

    def __call__(self, assignee, function, arg_dict,
            get_new_identifier, add_declaration):
        from leap.vm.codegen.utils import (
                chop_common_indentation,
                remove_redundant_blank_lines)

        args = function.resolve_args(arg_dict)
        return remove_redundant_blank_lines(
                chop_common_indentation(
                    self.template.render(
                        assignee=assignee,
                        get_new_identifier=get_new_identifier,
                        add_declaration=add_declaration,
                        **dict(zip(function.arg_names, args)))))


class FortranType(object):
    """
    .. attribute:: base_type
    .. attribute:: dimension

        A tuple of ``'200'``, ``'-5:5'``, or some such.
        Entries may be numeric, too.
    """

    def __init__(self, base_type, dimension):
        self.base_type = base_type
        if dimension:
            dimension = tuple(str(d) for d in dimension)
        self.dimension = dimension


# {{{ code generator

class _IRFunctionDescriptor(Record):
    """
    .. attribute:: name
    .. attribute:: function
    .. attribute:: control_tree
    """


class FortranCodeGenerator(StructuredCodeGenerator):
    language = "fortran"

    def __init__(self, module_name,
            ode_component_type_map,
            function_registry=None,
            module_preamble=None,
            real_scalar_kind="8",
            complex_scalar_kind="8",
            use_complex_scalars=True,
            trace=False):
        """
        :arg ode_component_type_map: a map from ODE component_id names
            to :class:`FortranType` instances
        """
        if function_registry is None:
            from leap.vm.function_registry import base_function_registry
            function_registry = base_function_registry

        self.module_name = module_name
        self.function_registry = function_registry
        self.ode_component_type_map = ode_component_type_map

        self.trace = trace

        from leap.vm.codegen.utils import chop_common_indentation
        self.module_preamble = chop_common_indentation(module_preamble)

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

    def emit_traceable(self, line):
        self.emit_trace(line)
        self.emit(line)

    def emit_trace(self, line):
        if self.trace:
            self.emit("write(*,*) '%s'" % line)

    def __call__(self, dag, optimize=True):
        from .analysis import verify_code
        verify_code(dag)

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

        for state_name, dependencies in six.iteritems(dag.states):
            code = dag_extractor(dag.instructions, dependencies)
            function = assembler(state_name, code, dependencies)
            if optimize:
                function = optimizer(function)
            control_tree = extract_structure(function)

            fdescrs.append(
                    _IRFunctionDescriptor(
                        name=state_name,
                        function=function,
                        control_tree=control_tree))

        # }}}

        from leap.vm.codegen.data import SymbolKindFinder

        self.sym_kind_table = SymbolKindFinder(self.function_registry)([
            fd.function for fd in fdescrs])

        from .analysis import collect_ode_component_names_from_dag
        component_ids = collect_ode_component_names_from_dag(dag)

        if not component_ids <= set(self.ode_component_type_map):
            raise RuntimeError("ODE components with undeclared types: %r"
                    % (component_ids - set(self.ode_component_type_map)))

        from leap.vm.codegen.data import Scalar, ODEComponent
        for comp_id in component_ids:
            self.sym_kind_table.set(
                    None, "<ret_time_id>"+comp_id, Scalar(is_real_valued=True))
            self.sym_kind_table.set(
                    None, "<ret_time>"+comp_id, Scalar(is_real_valued=True))
            self.sym_kind_table.set(
                    None, "<ret_state>"+comp_id, ODEComponent(comp_id))

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
        self.emit_def_end(function_name)

        self.current_function = None

    def begin_emit(self, dag):
        if self.module_preamble:
            for l in self.module_preamble:
                self.emit(l)
            self.emit('')

        from .analysis import collect_time_ids_from_dag
        for i, time_id in enumerate(collect_time_ids_from_dag(dag)):
            self.emit("parameter (leap_time_{time_id} = {i})".format(
                time_id=time_id, i=i))
        self.emit('')

        for i, state in enumerate(dag.states):
            self.emit("parameter (leap_state_{state_name} = {i})".format(
                state_name=state, i=i))

        self.emit('')

        with FortranTypeEmitter(
                self.emitter,
                'leap_state_type',
                self) as emit:
            emit('integer leap_state')
            emit('')

            for identifier, sym_kind in six.iteritems(
                    self.sym_kind_table.global_table):
                self.emit_variable_decl(
                        self.name_manager.name_global(identifier),
                        sym_kind)

        self.emit('contains')
        self.emit('')

    # {{{ data management

    def emit_variable_decl(self, sym, sym_kind,
            is_argument=False, other_specifiers=(),
            fortran_name=None):
        if fortran_name is None:
            fortran_name = self.name_manager[sym]

        from leap.vm.codegen.data import Boolean, Scalar

        type_specifiers = other_specifiers

        from leap.vm.codegen.data import ODEComponent
        if isinstance(sym_kind, Boolean):
            type_name = 'logical'

        elif isinstance(sym_kind, Scalar):
            if sym_kind.is_real_valued or not self.use_complex_scalars:
                type_name = 'real (kind=%s)' % self.real_scalar_kind
            else:
                type_name = 'complex (kind=%s)' % self.complex_scalar_kind

        elif isinstance(sym_kind, ODEComponent):
            comp_type = self.ode_component_type_map[sym_kind.component_id]
            type_name = comp_type.base_type

            if not is_argument:
                type_specifiers = type_specifiers + ('pointer',)
                self.emit(
                        'integer, pointer :: leap_refcnt_{id}'.format(
                            id=fortran_name))

            if comp_type.dimension:
                if is_argument:
                    type_specifiers += (
                            "dimension(%s)" % ",".join(comp_type.dimension),)
                else:
                    type_specifiers += (
                            "dimension(%s)"
                            % ",".join(len(comp_type.dimension)*":"),)

        else:
            raise ValueError("unknown variable kind: %s" % type(sym_kind).__name__)

        if type_specifiers:
            self.emit('{type_name}, {type_specifier_list} :: {id}'.format(
                type_name=type_name,
                type_specifier_list=", ".join(type_specifiers),
                id=fortran_name))
        else:
            self.emit('{type_name} {id}'.format(
                type_name=type_name,
                id=fortran_name))

    def emit_variable_init(self, name, sym_kind):
        fortran_name = self.name_manager[name]

        from leap.vm.codegen.data import ODEComponent
        if isinstance(sym_kind, ODEComponent):
            self.emit_traceable("nullify(%s)" % fortran_name)

    def emit_variable_deinit(self, name, sym_kind):
        fortran_name = self.name_manager[name]
        refcnt_name = self.name_manager.name_refcount(name)

        from leap.vm.codegen.data import ODEComponent
        if not isinstance(sym_kind, ODEComponent):
            return

        with FortranIfEmitter(
                self.emitter, 'associated(%s)' % fortran_name, self):
            with FortranIfEmitter(
                    self.emitter, '{refcnt}.eq.1'.format(refcnt=refcnt_name), self) \
                            as if_emit:
                self.emit_traceable('deallocate({id})'.format(id=fortran_name))
                self.emit_traceable(
                        'deallocate({refcnt})'.format(refcnt=refcnt_name))
                self.emit_traceable('nullify({id})'.format(id=fortran_name))

                if_emit.emit_else()

                self.emit_traceable(
                        '{refcnt} = {refcnt} - 1'
                        .format(refcnt=refcnt_name))

    def emit_allocation(self, sym, sym_kind):
        fortran_name = self.name_manager[sym]
        refcnt_name = self.name_manager.name_refcount(sym)

        from leap.vm.codegen.data import ODEComponent
        if not isinstance(sym_kind, ODEComponent):
            return

        comp_type = self.ode_component_type_map[sym_kind.component_id]

        if comp_type.dimension:
            dimension = '(%s)' % ', '.join(
                    str(dim_axis) for dim_axis in comp_type.dimension)
        else:
            dimension = ""

        from leap.vm.codegen.data import ODEComponent
        if not isinstance(sym_kind, ODEComponent):
            return

        self.emit_traceable('allocate({name}{dimension}, stat=leap_ierr)'.format(
            name=fortran_name,
            dimension=dimension))
        with FortranIfEmitter(self.emitter, 'leap_ierr.ne.0', self):
            self.emit("write(*,*) 'failed to allocate {name}'".format(
                name=fortran_name))
            self.emit("stop")

        self.emit_traceable('allocate({name}, stat=leap_ierr)'.format(
            name=refcnt_name))
        with FortranIfEmitter(self.emitter, 'leap_ierr.ne.0', self):
            self.emit("write(*,*) 'failed to allocate {name}'".format(
                name=refcnt_name))
            self.emit("stop")

        self.emit_traceable('{refcnt} = 1'.format(refcnt=refcnt_name))

    def emit_allocation_check(self, sym, sym_kind):
        fortran_name = self.name_manager[sym]
        refcnt_name = self.name_manager.name_refcount(sym)

        from leap.vm.codegen.data import ODEComponent
        if not isinstance(sym_kind, ODEComponent):
            return

        with FortranIfEmitter(
                self.emitter, '.not.associated(%s)' % fortran_name, self) as emit_if:
            self.emit_allocation(sym, sym_kind)

            emit_if.emit_else()

            with FortranIfEmitter(
                    self.emitter,
                    '{refcnt}.ne.1'.format(refcnt=refcnt_name),
                    self) as emit_if:
                self.emit_traceable(
                        '{refcnt} = {refcnt} - 1'
                        .format(refcnt=refcnt_name))

                self.emit('')

                self.emit_allocation(sym, sym_kind)

    # }}}

    def emit_initialize(self):
        init_symbols = [
                sym
                for sym in self.sym_kind_table.global_table
                if not sym.startswith("<ret")]

        args = ('leap_state',) + tuple(
                self.name_manager.name_global(sym)
                for sym in init_symbols)

        with FortranSubroutineEmitter(
                self.emitter,
                'initialize', args, self):

            self.emit('implicit none')
            self.emit('')
            self.emit('type(leap_state_type), pointer :: leap_state')
            self.emit('integer leap_ierr')
            self.emit('')

            for sym in init_symbols:
                sym_kind = self.sym_kind_table.global_table[sym]
                fortran_name = self.name_manager.name_global(sym)

                self.emit_variable_decl(
                        sym,
                        sym_kind, is_argument=True,
                        other_specifiers=("optional",),
                        fortran_name=fortran_name)
            self.emit('')

            for sym, sym_kind in six.iteritems(
                    self.sym_kind_table.global_table):
                self.emit_variable_init(sym, sym_kind)

            for sym in init_symbols:
                sym_kind = self.sym_kind_table.global_table[sym]

                tgt_fortran_name = self.name_manager[sym]
                fortran_name = self.name_manager.name_global(sym)

                with FortranIfEmitter(
                        self.emitter, 'present(%s)' % fortran_name, self):
                    self.emit_allocation(sym, sym_kind)

                    self.emit_traceable("{name} = {arg}"
                            .format(
                                name=tgt_fortran_name,
                                arg=fortran_name))

    def emit_shutdown(self):
        with FortranSubroutineEmitter(
                self.emitter,
                'shutdown', ('leap_state',), self):

            self.emit('implicit none')
            self.emit('')
            self.emit('type(leap_state_type), pointer :: leap_state')
            self.emit('')

            from leap.vm.codegen.data import ODEComponent

            for sym, sym_kind in six.iteritems(self.sym_kind_table.global_table):
                self.emit_variable_deinit(sym, sym_kind)

                if isinstance(sym_kind, ODEComponent):
                    fortran_name = self.name_manager[sym]
                    with FortranIfEmitter(
                            self.emitter,
                            'associated({id})'.format(id=fortran_name), self):
                        self.emit("write(*,*) 'leaked reference in {name}'".format(
                            name=fortran_name))
                        self.emit("write(*,*) '  remaining refcount ', {name}"
                                .format(name=self.name_manager.name_refcount(sym)))
                        self.emit('stop')

    def emit_run_step(self):
        """Emit the run() method."""
        with FortranSubroutineEmitter(
                self.emitter,
                'run', ('leap_state', 't_end'), self) as emit:
            emit('t_end = kwargs["t_end"]')
            emit('last_step = False')

            # STATE_HACK: This implementation of staging support should be replaced
            # so that the states are not hard-coded.
            emit('next_states = { "initialization": "primary", ' +
                 '"primary": "primary" }')
            emit('current_state = "initialization"')

            with FortranDoEmitter(emit, ".true.", self):
                with FortranIfEmitter(
                        self.emitter, 'self.t + self.dt >= t_end', self):
                    self.emit('assert self.t <= t_end')
                    self.emit('self.dt = t_end - self.t')
                    self.emit('last_step = True')

                d_emit('state_function = getattr(self, "state_" + current_state)')
                emit('result = state_function()')

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
                d_emit('current_state = next_states[current_state]')

    def finish_emit(self, dag):
        self.emit_initialize()
        self.emit_shutdown()
        # self.emit_run()

        self.module_emitter.__exit__(None, None, None)

    def get_code(self):
        return self.module_emitter.get()

    # {{{ called by superclass

    def emit_def_begin(self, func_id):
        self.declaration_emitter = FortranEmitter()

        FortranSubroutineEmitter(
                self.emitter,
                'leap_state_func_' + func_id, ('leap_state',),
                self).__enter__()

        self.declaration_emitter('implicit none')
        self.declaration_emitter('')
        self.declaration_emitter('type(leap_state_type), pointer :: leap_state')
        self.declaration_emitter('integer leap_ierr')
        self.declaration_emitter('')

        body_emitter = FortranEmitter()
        self.emitters.append(body_emitter)

        sym_table = self.sym_kind_table.per_function_table.get(func_id, {})
        for identifier, sym_kind in six.iteritems(sym_table):
            self.emit_variable_decl(identifier, sym_kind)

        if sym_table:
            self.emit('')

        self.emit_trace('================================================')
        self.emit_trace('enter %s' % func_id)

        for identifier, sym_kind in six.iteritems(sym_table):
            self.emit_variable_init(identifier, sym_kind)

        if sym_table:
            self.emit('')

    def emit_def_end(self, func_id):
        self.emitters[-2].incorporate(self.declaration_emitter)

        body_emitter = self.emitters.pop()
        self.emitter.incorporate(body_emitter)

        # body emitter
        self.emitter.__exit__(None, None, None)

        del self.declaration_emitter

    def emit_while_loop_begin(self, expr):
        self.emitter = FortranDoEmitter(
                self.emitter,
                self.expr(expr))
        self.emitter.__enter__()

    def emit_while_loop_end(self,):
        self.emitter.__exit__(None, None, None)

    def emit_if_begin(self, expr):
        self.emitter = FortranIfEmitter(
                self.emitter,
                self.expr(expr))

    emit_if_end = emit_while_loop_end

    def emit_else_begin(self):
        self.emitter.emit_else()

    emit_else_end = emit_while_loop_end

    def emit_assign_expr(self, assignee_sym, expr):
        from leap.vm.codegen.data import ODEComponent

        fortran_name = self.name_manager[assignee_sym]

        sym_kind = self.sym_kind_table.get(
                self.current_function, assignee_sym)

        if isinstance(sym_kind, ODEComponent):
            if isinstance(expr, Variable):
                self.emit_variable_deinit(assignee_sym, sym_kind)

                self.emit_traceable(
                    "{name} => {expr}"
                    .format(
                        name=fortran_name,
                        expr=self.name_manager[expr.name]))
                self.emit_traceable(
                    "{tgt_refcnt} => {refcnt}"
                    .format(
                        tgt_refcnt=self.name_manager.name_refcount(assignee_sym),
                        refcnt=self.name_manager.name_refcount(expr.name)))
                self.emit_traceable(
                    "{tgt_refcnt} = {tgt_refcnt} + 1"
                    .format(
                        tgt_refcnt=self.name_manager.name_refcount(assignee_sym)))
                self.emit('')
                return

            self.emit_allocation_check(assignee_sym, sym_kind)

        if isinstance(expr, (Call, CallWithKwargs)):
            self.emit_trace("rhs eval {assignee_sym} = {expr}..."
                    .format(
                        assignee_sym=assignee_sym,
                        expr=str(expr)[:50]))

            function = self.function_registry[expr.function.name]
            codegen = function.get_codegen(self.language)

            # TODO: get_new_identifier

            def add_declaration(decl):
                self.declaration_emitter.emit(decl)

            arg_strs_dict = {}
            for i, arg in enumerate(expr.parameters):
                arg_strs_dict[i] = "(%s)" % self.expr(arg)
            if isinstance(expr, CallWithKwargs):
                for arg_name, arg in expr.kw_parameters.items():
                    arg_strs_dict[arg_name] = "(%s)" % self.expr(arg)

            lines = codegen(
                    assignee=fortran_name,
                    function=function,
                    arg_dict=arg_strs_dict,
                    get_new_identifier=None,  # FIXME
                    add_declaration=add_declaration)

            for l in lines:
                self.emit(l)

        else:
            self.emit_trace("{assignee_sym} = {expr}..."
                    .format(
                        assignee_sym=assignee_sym,
                        expr=str(expr)[:50]))
            self.emit(
                    "{name} = {expr}"
                    .format(
                        name=fortran_name,
                        expr=self.expr(expr)))

        self.emit('')

    def emit_return(self):
        # {{{ emit variable deinit

        sym_table = self.sym_kind_table.per_function_table.get(
                self.current_function, {})

        for identifier, sym_kind in six.iteritems(sym_table):
            self.emit_variable_deinit(identifier, sym_kind)

        # }}}

        self.emit_trace('leave %s' % self.current_function)

        self.emit('return')

    def emit_yield_state(self, inst):
        self.emit_assign_expr(
                '<ret_time_id>'+inst.component_id,
                Variable("leap_time_"+str(inst.time_id)))
        self.emit_assign_expr(
                '<ret_time>'+inst.component_id,
                inst.time)
        self.emit_assign_expr(
                '<ret_state>'+inst.component_id,
                inst.expression)

    # }}}

# }}}

# vim: foldmethod=marker
