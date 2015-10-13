"""Fortran code generator"""

from __future__ import division

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

import sys
from functools import partial
import re  # noqa
import six

from .expressions import FortranExpressionMapper
from .codegen_base import StructuredCodeGenerator
from dagrt.vm.utils import is_state_variable
from pytools.py_codegen import (
        # It's the same code. So sue me.
        PythonCodeGenerator as FortranEmitterBase)
from pymbolic.primitives import (Call, CallWithKwargs, Variable,
        Subscript, Lookup)
from pymbolic.mapper import IdentityMapper
from .utils import wrap_line_base, KeyToUniqueNameMap


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
        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator()
        self.local_map = KeyToUniqueNameMap(name_generator=self.name_generator)
        self.global_map = KeyToUniqueNameMap(start={
                '<t>': 'leap_t', '<dt>': 'leap_dt'},
                name_generator=self.name_generator)
        self.function_map = KeyToUniqueNameMap(name_generator=self.name_generator)

    def name_global(self, var):
        """Return the identifier for a global variable."""
        return self.global_map.get_or_make_name_for_key(var)

    def name_local(self, var, prefix=None):
        """Return the identifier for a local variable."""
        if prefix is None:
            if not var.startswith("leap_"):
                prefix = "lploc_"

        return self.local_map.get_or_make_name_for_key(var, prefix=prefix)

    def name_function(self, var):
        """Return the identifier for a function."""
        return self.function_map.get_or_make_name_for_key(var)

    def make_unique_fortran_name(self, prefix):
        return self.local_map.get_mapped_identifier_without_key("lpfor_"+prefix)

    def is_known_fortran_name(self, name):
        return self.name_generator.is_name_conflicting(name)

    def __getitem__(self, name):
        """Provide an interface to the expression mapper to look up
        the name of a local or global variable.
        """
        if is_state_variable(name):
            return 'leap_state%'+self.name_global(name)
        else:
            return self.name_local(name)

    def name_refcount(self, name, qualified_with_state=True):
        if is_state_variable(name):
            if qualified_with_state:
                return 'leap_state%leap_refcnt_'+self.name_global(name)
            else:
                return 'leap_refcnt_'+self.name_global(name)
        else:
            return self.name_local("leap_refcnt_"+name)

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

    def emit_else_if(self, expr):
        self.dedent()
        self("else if ({expr}) then".format(expr=expr))
        self.indent()


class FortranDoEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, loop_var, bounds, code_generator=None):
        super(FortranDoEmitter, self).__init__(
                parent_emitter, "do", code_generator)
        self("do {loop_var} = {bounds}".format(
            loop_var=loop_var, bounds=bounds))


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


class CallCode(object):
    def __init__(self, template, extra_args=None):
        """
        :arg extra_args: a dictionary of names that should be made available
            in template expansion.
        """

        from mako.template import Template

        self.template = Template(template, strict_undefined=True)

        self.extra_args = extra_args

    def __call__(self, results, function, arg_strings_dict, arg_kinds_dict,
            code_generator):
        from dagrt.vm.codegen.utils import (
                remove_common_indentation,
                remove_redundant_blank_lines)

        args = function.resolve_args(arg_strings_dict)
        arg_kinds = function.resolve_args(arg_kinds_dict)

        def add_declaration(decl):
            code_generator.declaration_emitter(decl)

        def declare_new(decl_without_name, prefix):
            new_name = code_generator.name_manager.make_unique_fortran_name(prefix)
            code_generator.declaration_emitter(decl_without_name + " :: " + new_name)
            return new_name

        import dagrt.vm.codegen.data as kinds

        template_names = dict(
                real_scalar_kind=code_generator.real_scalar_kind,
                complex_scalar_kind=code_generator.complex_scalar_kind,
                get_new_identifier=(
                    code_generator.name_manager.make_unique_fortran_name),
                add_declaration=add_declaration,
                declare_new=declare_new,
                kinds=kinds)

        result_names = getattr(function, "result_names", ("result",))

        assert len(result_names) == len(results)
        for res_name, res in zip(result_names, results):
            template_names[res_name] = res

        template_names.update(zip(function.arg_names, args))

        template_names.update(
                (name+"_kind", kind)
                for name, kind in zip(function.arg_names, arg_kinds))

        if self.extra_args:
            template_names.update(self.extra_args)

        rendered = self.template.render(**template_names)

        if sys.version_info < (3,):
            rendered = rendered.encode()

        lines = remove_redundant_blank_lines(
                remove_common_indentation(rendered))

        for l in lines:
            code_generator.emit(l)


# {{{ expression modifiers

class ODEComponentReferenceTransformer(IdentityMapper):
    def __init__(self, sym_kind_table, current_function):
        self.sym_kind_table = sym_kind_table
        self.current_function = current_function

    def find_sym_kind(self, expr):
        if isinstance(expr, (Subscript, Lookup)):
            return self.find_sym_kind(expr.aggregate)
        elif isinstance(expr, Variable):
            return self.sym_kind_table.get(self.current_function, expr.name)
        else:
            raise TypeError("unsupported object")

    def map_variable(self, expr):
        from dagrt.vm.codegen.data import ODEComponent
        if isinstance(self.find_sym_kind(expr), ODEComponent):
            return self.transform(expr)
        else:
            return expr

    map_lookup = map_variable
    map_subscript = map_variable


class StructureLookupAppender(ODEComponentReferenceTransformer):
    def __init__(self, sym_kind_table, current_function, component):
        super(StructureLookupAppender, self).__init__(
                sym_kind_table, current_function)
        self.component = component

    def transform(self, expr):
        return expr.attr(self.component)


class ArraySubscriptAppender(ODEComponentReferenceTransformer):
    def __init__(self, sym_kind_table, current_function, subscript):
        super(ArraySubscriptAppender, self).__init__(
                sym_kind_table, current_function)
        self.subscript = subscript

    def transform(self, expr):
        return expr.index(self.subscript)

# }}}


# {{{ types

class TypeBase(object):
    """
    .. attribute:: base_type
    .. attribute:: dimension

        A tuple of ``'200'``, ``'-5:5'``, or some such.
        Entries may be numeric, too.
    """

    def get_base_type(self):
        raise NotImplementedError()

    def get_type_specifiers(self, defer_dim):
        raise NotImplementedError()

    def is_allocatable(self):
        raise NotImplementedError()


class BuiltinType(TypeBase):
    def __init__(self, type_name):
        self.type_name = type_name

    def get_base_type(self):
        return self.type_name

    def get_type_specifiers(self, defer_dim):
        return ()

    def is_allocatable(self):
        return False


# Allocatable arrays are not yet supported, use pointers for now.
class ArrayType(TypeBase):
    """
    .. attribute:: dimension

        A tuple of ``'200'``, ``'-5:5'``, or some such.
        Entries may be numeric, too. Or they may refer
        to variables that are available through
        *extra_arguments* in :class:`CodeGenerator`.

    .. attribute:: index_vars

        If further dimensions within :attr:`element_type` depend
        on indices into :attr:`dimension`, this tuple of variable
        names determines what each index variable is called.
    """

    @staticmethod
    def parse_dimension(dim):
        parts = ":".split(dim)

        if len(parts) == 1:
            return ("1", dim)
        elif len(parts) == 2:
            return tuple(parts)
        else:
            raise RuntimeError(
                    "unexpected number of parts in dimension spec '%s'"
                    % dim)

    INDEX_VAR_COUNTER = 0

    def __init__(self, dimension, element_type, index_vars=None):
        self.element_type = element_type
        if isinstance(dimension, str):
            dimension = tuple(d.strip() for d in dimension.split(","))
        self.dimension = tuple(str(i) for i in dimension)

        if isinstance(index_vars, str):
            index_vars = tuple(iv.strip() for iv in index_vars.split(","))
        elif index_vars is None:
            def get_index_var():
                ArrayType.INDEX_VAR_COUNTER += 1
                return "i%d" % ArrayType.INDEX_VAR_COUNTER

            index_vars = tuple(get_index_var() for d in dimension)

        if len(index_vars) != len(dimension):
            raise ValueError("length of 'index_vars' does not match length "
                    "of 'dimension'")

        if not isinstance(element_type, TypeBase):
            raise TypeError("element_type should be a subclass of TypeBase")
        if isinstance(element_type, PointerType):
            raise TypeError("Arrays of pointers are not allowed in Fortran. "
                    "You must declare an intermediate StructureType instead.")

        self.index_vars = index_vars

    def get_base_type(self):
        return self.element_type.get_base_type()

    def get_type_specifiers(self, defer_dim):
        result = self.element_type.get_type_specifiers(defer_dim)

        if defer_dim:
            result += (
                    "dimension(%s)" % ",".join(len(self.dimension)*":"),)
        else:
            result += (
                    "dimension(%s)" % ",".join(self.dimension),)

        return result

    def is_allocatable(self):
        return self.element_type.is_allocatable()


class PointerType(TypeBase):
    def __init__(self, pointee_type):
        self.pointee_type = pointee_type

        if not isinstance(pointee_type, TypeBase):
            raise TypeError("pointee_type should be a subclass of TypeBase")

    def get_base_type(self):
        return self.pointee_type.get_base_type()

    def get_type_specifiers(self, defer_dim):
        return self.pointee_type.get_type_specifiers(defer_dim=True) + ("pointer",)

    def is_allocatable(self):
        return True


class StructureType(TypeBase):
    """
    .. attribute:: members

        A tuple of **(name, type)** tuples.
    """

    def __init__(self, type_name, members):
        self.type_name = type_name
        self.members = members

        for i, (_, mtype) in enumerate(members):
            if not isinstance(mtype, TypeBase):
                raise TypeError("member with index %d has type that is "
                        "not a subclass of TypeBase"
                        % i)

    def get_base_type(self):
        return "type(%s)" % self.type_name

    def get_type_specifiers(self, defer_dim):
        return ()

    def is_allocatable(self):
        return any(
                member_type.is_allocatable()
                for name, member_type in self.members)

# }}}


# {{{ type visitor

# {{{ helper functionality

def _replace_indices(index_expr_map, s):
    for name, expr in six.iteritems(index_expr_map):
        s, _ = re.subn(r"\b" + name + r"\b", expr, s)
    return s


class _ArrayLoopManager(object):
    def __init__(self, array_type):
        self.array_type = array_type

    def enter(self, code_generator, index_expr_map):
        atype = self.array_type

        index_expr_map = index_expr_map.copy()

        f_index_names = [
            code_generator.name_manager.make_unique_fortran_name(iname)
            for iname in atype.index_vars]

        self.emitters = []
        for i, (dim, index_name) in enumerate(
                reversed(list(zip(atype.dimension, f_index_names)))):
            code_generator.declaration_emitter('integer %s' % index_name)

            start, stop = atype.parse_dimension(dim)
            em = FortranDoEmitter(
                    code_generator.emitter, index_name,
                    "%s, %s" % (
                        _replace_indices(index_expr_map, start),
                        _replace_indices(index_expr_map, stop)),
                    code_generator)
            self.emitters.append(em)
            em.__enter__()

        index_expr_map.update(zip(atype.index_vars, f_index_names))

        from pymbolic import var

        asa = ArraySubscriptAppender(
                code_generator.sym_kind_table, code_generator.current_function,
                tuple(var("<target>"+v) for v in f_index_names))

        return index_expr_map, asa, f_index_names

    def leave(self):
        while self.emitters:
            em = self.emitters.pop()
            em.__exit__(None, None, None)

# }}}


class TypeVisitor(object):
    recurse_only_if_allocatable = False

    def __init__(self, code_generator):
        self.code_generator = code_generator

    def rec(self, fortran_type, fortran_expr, index_expr_map, *args):
        return getattr(self, "visit_"+type(fortran_type).__name__)(
                fortran_type, fortran_expr, index_expr_map, *args)

    __call__ = rec

    def visit_BuiltinType(self, fortran_type, fortran_expr, index_expr_map,
            *args):
        pass

    def visit_ArrayType(self, fortran_type, fortran_expr, index_expr_map,
            *args):
        if (self.recurse_only_if_allocatable
                and not fortran_type.element_type.is_allocatable()):
            return

        alm = _ArrayLoopManager(fortran_type)
        index_expr_map, _, f_index_names = \
                alm.enter(self.code_generator, index_expr_map)

        self.rec(fortran_type.element_type,
                "%s(%s)" % (fortran_expr, ", ".join(f_index_names)),
                index_expr_map, *args)

        alm.leave()

    def visit_PointerType(self, fortran_type, fortran_expr, index_expr_map,
            *args):
        if (self.recurse_only_if_allocatable
                and not fortran_type.pointee_type.is_allocatable()):
            return

        self.rec(fortran_type.pointee_type, fortran_expr, index_expr_map,
                *args)

    def visit_StructureType(self, fortran_type, fortran_expr, index_expr_map,
            *args):
        for member_name, member_type in fortran_type.members:
            if (self.recurse_only_if_allocatable
                    and not member_type.is_allocatable()):
                continue

            self.rec(member_type,
                    fortran_expr+"%"+member_name,
                    index_expr_map,
                    *args)


class AssignmentEmitter(TypeVisitor):
    def visit_BuiltinType(self, fortran_type, fortran_expr, index_expr_map,
            rhs_expr):
        self.code_generator.emit(
                "{name} = {expr}"
                .format(
                    name=fortran_expr,
                    expr=self.code_generator.expr(rhs_expr)))

    def visit_ArrayType(self, fortran_type, fortran_expr, index_expr_map,
            rhs_expr):
        alm = _ArrayLoopManager(fortran_type)
        index_expr_map, array_subscript_appender, f_index_names = \
                alm.enter(self.code_generator, index_expr_map)

        self.rec(fortran_type.element_type,
                "%s(%s)" % (fortran_expr, ", ".join(f_index_names)),
                index_expr_map, array_subscript_appender(rhs_expr))

        alm.leave()

    def visit_StructureType(self, fortran_type, fortran_expr, index_expr_map,
            rhs_expr):
        for member_name, member_type in fortran_type.members:
            sla = StructureLookupAppender(
                    self.code_generator.sym_kind_table,
                    self.code_generator.current_function,
                    member_name)

            self.rec(member_type,
                    fortran_expr+"%"+member_name,
                    index_expr_map,
                    sla(rhs_expr))


class AllocationEmitter(TypeVisitor):
    recurse_only_if_allocatable = True

    def visit_PointerType(self, fortran_type, fortran_expr, index_expr_map):
        pointee_type = fortran_type.pointee_type
        code_generator = self.code_generator

        dimension = ""
        if isinstance(pointee_type, ArrayType):
            if pointee_type.dimension:
                dimension = '(%s)' % ', '.join(
                        str(dim_axis) for dim_axis in pointee_type.dimension)

        code_generator.emit_traceable(
                'allocate({name}{dimension}, stat=leap_ierr)'.format(
                    name=fortran_expr,
                    dimension=_replace_indices(index_expr_map, dimension)))
        with FortranIfEmitter(
                code_generator.emitter, 'leap_ierr.ne.0', code_generator):
            code_generator.emit(
                    "write(leap_stderr,*) 'failed to allocate {name}'".format(
                        name=fortran_expr))
            code_generator.emit("stop")

        if pointee_type.is_allocatable():
            self.rec(pointee_type, fortran_expr, index_expr_map)


class DeallocationEmitter(TypeVisitor):
    recurse_only_if_allocatable = True

    def __init__(self, code_generator, deinitializer):
        super(DeallocationEmitter, self).__init__(code_generator)
        self.deinitializer = deinitializer

    def visit_PointerType(self, fortran_type, fortran_expr, index_expr_map):
        pointee_type = fortran_type.pointee_type
        code_generator = self.code_generator

        if pointee_type.is_allocatable():
            self.rec(pointee_type, fortran_expr, index_expr_map)
        self.deinitializer(pointee_type, fortran_expr, index_expr_map)

        code_generator.emit_traceable('deallocate({id})'.format(id=fortran_expr))
        code_generator.emit_traceable("nullify(%s)" % fortran_expr)


class InitializationEmitter(TypeVisitor):
    recurse_only_if_allocatable = True

    def visit_PointerType(self, fortran_type, fortran_expr, index_expr_map):
        self.code_generator.emit_traceable("nullify(%s)" % fortran_expr)

# }}}


# {{{ code generator

class CodeGenerator(StructuredCodeGenerator):
    language = "fortran"

    def __init__(self, module_name,
            ode_component_type_map,
            function_registry=None,
            module_preamble=None,
            real_scalar_kind="8",
            complex_scalar_kind="8",
            use_complex_scalars=True,
            call_before_state_update=None,
            call_after_state_update=None,
            extra_arguments=(),
            extra_argument_decl=None,
            trace=False):
        """
        :arg function_registry:
        :arg ode_component_type_map: a map from ODE component_id names
            to :class:`FortranType` instances
        :arg call_before_state_update: The name of a function that should
            be called before each state update. The function must be known
            to *function_registry*.
        :arg call_after_state_update: The name of a function that should
            be called after each state update. The function must be known
            to *function_registry*.
        :arg extra_arguments: A tuple of names of extra arguments that are
            prepended to the call signature of each generated function
            and are available to refer to in user-supplied function
            implementations.
        :arg extra_argument_decl: Type declarations for *extra_arguments*,
            inserted into each generated function. Must be a multi-line
            string whose first line is empty, typically from a triple-quoted
            string in Python. Leading indentation is removed.
        """
        if function_registry is None:
            from dagrt.vm.function_registry import base_function_registry
            function_registry = base_function_registry

        self.module_name = module_name
        self.function_registry = function_registry
        self.ode_component_type_map = ode_component_type_map

        self.trace = trace

        from dagrt.vm.codegen.utils import remove_common_indentation
        self.module_preamble = remove_common_indentation(module_preamble)

        self.real_scalar_kind = real_scalar_kind
        self.complex_scalar_kind = complex_scalar_kind
        self.use_complex_scalars = use_complex_scalars
        self.call_before_state_update = call_before_state_update
        self.call_after_state_update = call_after_state_update

        if isinstance(extra_arguments, str):
            extra_arguments = tuple(s.strip() for s in extra_arguments.split(","))

        self.extra_arguments = extra_arguments
        if extra_argument_decl is not None:
            from dagrt.vm.codegen.utils import remove_common_indentation
            extra_argument_decl = remove_common_indentation(
                extra_argument_decl)
        self.extra_argument_decl = extra_argument_decl

        self.name_manager = FortranNameManager()
        self.expr_mapper = FortranExpressionMapper(
                self.name_manager)

        # FIXME: Should make extra arguments known to
        # name manager

        self.module_emitter = FortranModuleEmitter(module_name)
        self.module_emitter.__enter__()

        self.emitters = [self.module_emitter]

    @staticmethod
    def state_name_to_state_sym(state_name):
        return "leap_state_"+state_name

    @staticmethod
    def component_name_to_component_sym(comp_name):
        return "leap_component_"+comp_name

    @property
    def emitter(self):
        return self.emitters[-1]

    def expr(self, expr):
        return self.expr_mapper(expr)

    def rhs(self, rhs):
        return self.name_manager.name_rhs(rhs)

    def emit(self, line):
        self.emitter(line)

    def emit_traceable(self, line):
        self.emit_trace(line)
        self.emit(line)

    def emit_trace(self, line):
        if self.trace:
            self.emit("write(*,*) '%s'" % line)

    def __call__(self, dag, optimize=True):
        from .analysis import verify_code
        verify_code(dag)

        from .transform import (
                eliminate_self_dependencies,
                isolate_function_arguments,
                isolate_function_calls,
                expand_IfThenElse)
        dag = eliminate_self_dependencies(dag)
        dag = isolate_function_arguments(dag)
        dag = isolate_function_calls(dag)
        dag = expand_IfThenElse(dag)

        # from dagrt.vm.language import show_dependency_graph
        # show_dependency_graph(dag)

        # {{{ produce function name / function AST pairs

        from .ast_ import create_ast_from_state

        from collections import namedtuple
        NameASTPair = namedtuple("NameASTPair", "name, ast")  # noqa
        fdescrs = []

        for state_name in six.iterkeys(dag.states):
            ast = create_ast_from_state(dag, state_name, optimize)
            fdescrs.append(NameASTPair(state_name, ast))

        # }}}

        from dagrt.vm.codegen.data import SymbolKindFinder

        self.sym_kind_table = SymbolKindFinder(self.function_registry)(
            [fd.name for fd in fdescrs],
            [fd.ast for fd in fdescrs])

        from .analysis import collect_ode_component_names_from_dag
        component_ids = collect_ode_component_names_from_dag(dag)

        if not component_ids <= set(self.ode_component_type_map):
            raise RuntimeError("ODE components with undeclared types: %r"
                    % (component_ids - set(self.ode_component_type_map)))

        from dagrt.vm.codegen.data import Scalar, ODEComponent
        for comp_id in component_ids:
            self.sym_kind_table.set(
                    None, "<ret_time_id>"+comp_id, Scalar(is_real_valued=True))
            self.sym_kind_table.set(
                    None, "<ret_time>"+comp_id, Scalar(is_real_valued=True))
            self.sym_kind_table.set(
                    None, "<ret_state>"+comp_id, ODEComponent(comp_id))

        self.begin_emit(dag)
        for fdescr in fdescrs:
            self.lower_function(fdescr.name, fdescr.ast)
        self.finish_emit(dag)

        del self.sym_kind_table

        code = self.get_code()

        new_lines = []
        for l in code.split("\n"):
            if l.lstrip().startswith("#"):
                hashmark_pos = l.find("#")
                assert hashmark_pos >= 0
                l = "#" + l[:hashmark_pos] + l[hashmark_pos+1:]

            new_lines.append(l)

        return "\n".join(new_lines)

    def lower_function(self, function_name, ast):
        self.current_function = function_name

        self.emit_def_begin(
                'leap_state_func_' + function_name,
                self.extra_arguments + ('leap_state',),
                state_id=function_name)
        self.declaration_emitter('type(leap_state_type), pointer :: leap_state')
        self.declaration_emitter('')

        self.lower_ast(ast)
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
            self.emit("parameter ({state_sym_name} = {i})".format(
                state_sym_name=self.state_name_to_state_sym(state), i=i))

        self.emit('')

        from .analysis import collect_ode_component_names_from_dag
        component_ids = collect_ode_component_names_from_dag(dag)

        for i, comp_id in enumerate(component_ids):
            self.emit("parameter ({comp_sym_name} = {i})".format(
                comp_sym_name=self.component_name_to_component_sym(comp_id),
                i=i))

        self.emit('')

        with FortranTypeEmitter(
                self.emitter,
                'leap_state_type',
                self) as emit:
            emit('integer leap_next_state')
            emit('')

            for identifier, sym_kind in six.iteritems(
                    self.sym_kind_table.global_table):
                self.emit_variable_decl(
                        identifier,
                        self.name_manager.name_global(identifier),
                        sym_kind=sym_kind)

        self.emit('integer leap_stderr')
        self.emit('parameter (leap_stderr=0)')
        self.emit('')
        self.emit('contains')
        self.emit('')

    # {{{ data management

    def get_ode_component_type(self, component_id, is_argument=False):
        comp_type = self.ode_component_type_map[component_id]
        if not is_argument:
            comp_type = PointerType(comp_type)

        return comp_type

    def emit_variable_decl(self, sym, fortran_name, sym_kind,
            is_argument=False, other_specifiers=(), emit=None):
        if emit is None:
            emit = self.emit

        from dagrt.vm.codegen.data import Boolean, Scalar, Array, Integer

        type_specifiers = other_specifiers

        from dagrt.vm.codegen.data import ODEComponent
        if isinstance(sym_kind, Boolean):
            type_name = 'logical'

        elif isinstance(sym_kind, Array):
            if sym_kind.is_real_valued or not self.use_complex_scalars:
                type_name = 'real (kind=%s)' % self.real_scalar_kind
            else:
                type_name = 'complex (kind=%s)' % self.complex_scalar_kind
            type_specifiers = type_specifiers + ("allocatable", "dimension(:)")

        elif isinstance(sym_kind, Scalar):
            if sym_kind.is_real_valued or not self.use_complex_scalars:
                type_name = 'real (kind=%s)' % self.real_scalar_kind
            else:
                type_name = 'complex (kind=%s)' % self.complex_scalar_kind

        elif isinstance(sym_kind, Integer):
            type_name = 'integer'

        elif isinstance(sym_kind, ODEComponent):
            comp_type = self.get_ode_component_type(sym_kind.component_id,
                    is_argument=is_argument)

            type_name = comp_type.get_base_type()
            type_specifiers = (
                    type_specifiers
                    + comp_type.get_type_specifiers(defer_dim=is_argument))

            if not is_argument:
                emit(
                        'integer, pointer ::  {refcnt_name}'.format(
                            refcnt_name=self.name_manager.name_refcount(
                                sym, qualified_with_state=False)))

        else:
            raise ValueError("unknown variable kind: %s" % type(sym_kind).__name__)

        if type_specifiers:
            emit('{type_name}, {type_specifier_list} :: {id}'.format(
                type_name=type_name,
                type_specifier_list=", ".join(type_specifiers),
                id=fortran_name))
        else:
            emit('{type_name} {id}'.format(
                type_name=type_name,
                id=fortran_name))

    def emit_variable_init(self, name, sym_kind):
        from dagrt.vm.codegen.data import ODEComponent
        if isinstance(sym_kind, ODEComponent):
            comp_type = self.get_ode_component_type(sym_kind.component_id)
            InitializationEmitter(self)(comp_type, self.name_manager[name], {})

    def emit_variable_deinit(self, name, sym_kind):
        fortran_name = self.name_manager[name]
        refcnt_name = self.name_manager.name_refcount(name)

        from dagrt.vm.codegen.data import ODEComponent
        if not isinstance(sym_kind, ODEComponent):
            return

        with FortranIfEmitter(
                self.emitter, 'associated(%s)' % fortran_name, self):
            with FortranIfEmitter(
                    self.emitter, '{refcnt}.eq.1'.format(refcnt=refcnt_name), self) \
                            as if_emit:
                comp_type = self.get_ode_component_type(sym_kind.component_id)
                DeallocationEmitter(self, InitializationEmitter(self))(
                        comp_type, self.name_manager[name], {})

                self.emit_traceable(
                        'deallocate({refcnt})'.format(refcnt=refcnt_name))

                if_emit.emit_else()

                InitializationEmitter(self)(comp_type, self.name_manager[name], {})
                self.emit_traceable(
                        '{refcnt} = {refcnt} - 1'
                        .format(refcnt=refcnt_name))

    def emit_refcounted_allocation(self, sym, sym_kind):
        fortran_name = self.name_manager[sym]

        from dagrt.vm.codegen.data import ODEComponent
        if not isinstance(sym_kind, ODEComponent):
            return

        self.emit_allocation(fortran_name, sym_kind)
        self.emit_allocate_refcount(sym)

    def emit_allocate_refcount(self, sym):
        refcnt_name = self.name_manager.name_refcount(sym)

        self.emit_traceable('allocate({name}, stat=leap_ierr)'.format(
            name=refcnt_name))
        with FortranIfEmitter(self.emitter, 'leap_ierr.ne.0', self):
            self.emit("write(leap_stderr,*) 'failed to allocate {name}'".format(
                name=refcnt_name))
            self.emit("stop")

        self.emit_traceable('{refcnt} = 1'.format(refcnt=refcnt_name))

    def emit_allocation(self, fortran_name, sym_kind):
        comp_type = self.get_ode_component_type(sym_kind.component_id)
        AllocationEmitter(self)(comp_type, fortran_name, {})

    def emit_allocation_check(self, sym, sym_kind):
        fortran_name = self.name_manager[sym]
        refcnt_name = self.name_manager.name_refcount(sym)

        from dagrt.vm.codegen.data import ODEComponent
        assert isinstance(sym_kind, ODEComponent)

        with FortranIfEmitter(
                self.emitter, '.not.associated(%s)' % fortran_name, self) as emit_if:
            self.emit_refcounted_allocation(sym, sym_kind)

            emit_if.emit_else()

            with FortranIfEmitter(
                    self.emitter,
                    '{refcnt}.ne.1'.format(refcnt=refcnt_name),
                    self) as emit_if:
                self.emit_traceable(
                        '{refcnt} = {refcnt} - 1'
                        .format(refcnt=refcnt_name))

                self.emit('')

                self.emit_refcounted_allocation(sym, sym_kind)

    def emit_ode_component_move(self, assignee_sym, assignee_fortran_name,
            sym_kind, expr):
        self.emit_variable_deinit(assignee_sym, sym_kind)

        self.emit_traceable(
            "{name} => {expr}"
            .format(
                name=assignee_fortran_name,
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

    def emit_assign_expr_inner(self,
            assignee_fortran_name, assignee_subscript, expr, sym_kind):
        if assignee_subscript:
            subscript_str = "(%s)" % (
                    ", ".join(
                        "int(%s)" % self.expr(i)
                        for i in assignee_subscript))
        else:
            subscript_str = ""

        if isinstance(expr, (Call, CallWithKwargs)):
            # These are supposed to have been transformed to AssignFunctionCall.
            raise RuntimeError("bare Call/CallWithKwargs encountered in "
                    "Fortran code generator")

        else:
            self.emit_trace("{assignee_fortran_name}{subscript_str} = {expr}..."
                    .format(
                        assignee_fortran_name=assignee_fortran_name,
                        subscript_str=subscript_str,
                        expr=str(expr)[:50]))

            from dagrt.vm.codegen.data import ODEComponent
            if not isinstance(sym_kind, ODEComponent):
                self.emit(
                        "{name}{subscript_str} = {expr}"
                        .format(
                            name=assignee_fortran_name,
                            subscript_str=subscript_str,
                            expr=self.expr(expr)))
            else:
                comp_type = self.get_ode_component_type(sym_kind.component_id)
                AssignmentEmitter(self)(comp_type, assignee_fortran_name, {}, expr)

        self.emit('')

    def emit_extra_arg_decl(self, emitter=None):
        if emitter is None:
            emitter = self.emit

        if not self.extra_argument_decl:
            return

        for l in self.extra_argument_decl:
            emitter(l)

        self.emit('')

    # }}}

    def emit_initialize(self, dag):
        init_symbols = [
                sym
                for sym in self.sym_kind_table.global_table
                if not sym.startswith("<ret")]

        args = self.extra_arguments + ('leap_state',) + tuple(
                self.name_manager.name_global(sym)
                for sym in init_symbols)

        function_name = 'initialize'
        state_id = "<leap>"+function_name
        self.emit_def_begin(function_name, args, state_id=state_id)

        for sym in init_symbols:
            sym_kind = self.sym_kind_table.global_table[sym]
            fortran_name = self.name_manager.name_global(sym)
            self.sym_kind_table.set(state_id, "<target>"+fortran_name, sym_kind)

        self.declaration_emitter('type(leap_state_type), pointer :: leap_state')
        self.declaration_emitter('')

        self.current_function = state_id

        for sym in init_symbols:
            sym_kind = self.sym_kind_table.global_table[sym]
            fortran_name = self.name_manager.name_global(sym)

            self.emit_variable_decl(
                    sym,
                    fortran_name, sym_kind, is_argument=True,
                    other_specifiers=("optional",))
        self.emit('')

        self.emit('real :: leap_nan, leap_nan_helper')
        self.emit('')
        self.emit('leap_nan_helper = 0.0')
        self.emit('leap_nan = 0.0/leap_nan_helper')
        self.emit('')

        self.emit(
                "leap_state%leap_next_state = leap_state_{0}"
                .format(dag.initial_state))

        for sym, sym_kind in six.iteritems(
                self.sym_kind_table.global_table):
            self.emit_variable_init(sym, sym_kind)

        # {{{ initialize scalar outputs to NaN

        self.emit('')
        self.emit('! initialize scalar outputs to NaN')
        self.emit('')

        for sym in self.sym_kind_table.global_table:
            sym_kind = self.sym_kind_table.global_table[sym]

            tgt_fortran_name = self.name_manager[sym]

            # All our scalars are floating-point numbers for now,
            # so initializing them all to NaN is fine.

            from dagrt.vm.codegen.data import Scalar
            if sym.startswith("<ret") and isinstance(sym_kind, Scalar):
                self.emit('{fortran_name} = leap_nan'.format(
                    fortran_name=tgt_fortran_name))

        self.emit('')

        # }}}

        for sym in init_symbols:
            sym_kind = self.sym_kind_table.global_table[sym]

            tgt_fortran_name = self.name_manager[sym]
            fortran_name = self.name_manager.name_global(sym)

            with FortranIfEmitter(
                    self.emitter, 'present(%s)' % fortran_name, self):
                self.emit_refcounted_allocation(sym, sym_kind)

                from dagrt.vm.codegen.data import ODEComponent
                if not isinstance(sym_kind, ODEComponent):
                    self.emit(
                            "{lhs} = {rhs}"
                            .format(
                                lhs=tgt_fortran_name,
                                rhs=fortran_name))
                else:
                    comp_type = self.get_ode_component_type(
                            sym_kind.component_id)

                    from pymbolic import var
                    AssignmentEmitter(self)(comp_type, tgt_fortran_name, {},
                            var("<target>"+fortran_name))

        self.emit_def_end(function_name)

        self.current_function = None

    def emit_shutdown(self):
        args = self.extra_arguments + ('leap_state',)

        function_name = 'shutdown'
        state_id = "<leap>"+function_name
        self.emit_def_begin(function_name, args, state_id=state_id)

        self.declaration_emitter('type(leap_state_type), pointer :: leap_state')
        self.declaration_emitter('')

        self.current_function = state_id

        from dagrt.vm.codegen.data import ODEComponent

        for sym, sym_kind in six.iteritems(self.sym_kind_table.global_table):
            self.emit_variable_deinit(sym, sym_kind)

        for sym, sym_kind in six.iteritems(self.sym_kind_table.global_table):
            if isinstance(sym_kind, ODEComponent):
                fortran_name = self.name_manager[sym]
                with FortranIfEmitter(
                        self.emitter,
                        'associated({id})'.format(id=fortran_name), self):
                    self.emit(
                            "write(leap_stderr,*) 'leaked reference in {name}'"
                            .format(name=fortran_name))
                    self.emit(
                            "write(leap_stderr,*) 'remaining refcount ', {name}"
                            .format(name=self.name_manager.name_refcount(sym)))

        self.emit_def_end(function_name)

        self.current_function = None

    def emit_run_step(self, dag):
        args = self.extra_arguments + ('leap_state',)

        function_name = 'run'
        state_id = "<leap>"+function_name
        self.emit_def_begin(function_name, args, state_id=state_id)

        self.declaration_emitter('type(leap_state_type), pointer :: leap_state')
        self.declaration_emitter('')

        self.current_function = state_id

        if_emit = None
        for name, state_descr in six.iteritems(dag.states):
            state_sym_name = self.state_name_to_state_sym(name)
            cond = "leap_state%leap_next_state == "+state_sym_name

            if if_emit is None:
                if_emit = FortranIfEmitter(
                        self.emitter, cond, self)
                if_emit.__enter__()
            else:
                if_emit.emit_else_if(cond)

            self.emit(
                    "leap_state%leap_next_state = "
                    + self.state_name_to_state_sym(state_descr.next_state))

            self.emit(
                    "call leap_state_func_{state_name}({args})".format(
                        state_name=name,
                        args=", ".join(args)))

        if if_emit:
            if_emit.emit_else()
            self.emit("write(leap_stderr,*) 'encountered invalid state in run', "
                    "leap_state%leap_next_state")
            self.emit("stop")

        if_emit.__exit__(None, None, None)

        self.emit_def_end(function_name)

        self.current_function = None

    def finish_emit(self, dag):
        self.emit_initialize(dag)
        self.emit_shutdown()
        self.emit_run_step(dag)

        self.module_emitter.__exit__(None, None, None)

    def get_code(self):
        assert not self.module_emitter.preamble

        indent_spaces = 1
        indentation = indent_spaces*' '

        wrapped_lines = []
        for l in self.module_emitter.code:
            line_leading_spaces = (len(l) - len(l.lstrip(" ")))
            level = line_leading_spaces // indent_spaces
            line_ind = level*indentation
            if l[line_leading_spaces:].startswith("!"):
                wrapped_lines.append(l)
            else:
                for wrapped_line in wrap_line(
                        l[line_leading_spaces:],
                        level, indentation=indentation):
                    wrapped_lines.append(line_ind+wrapped_line)

        return "\n".join(wrapped_lines)

    # {{{ called by superclass

    def emit_def_begin(self, function_name, argument_names, state_id=None):
        self.declaration_emitter = FortranEmitter()

        FortranSubroutineEmitter(
                self.emitter,
                function_name,
                argument_names,
                self).__enter__()

        self.declaration_emitter('implicit none')
        self.declaration_emitter('integer leap_ierr')
        self.declaration_emitter('')

        self.emit_extra_arg_decl(self.declaration_emitter)

        body_emitter = FortranEmitter()
        self.emitters.append(body_emitter)

        if state_id is not None:
            sym_table = self.sym_kind_table.per_function_table.get(state_id, {})
            for identifier, sym_kind in six.iteritems(sym_table):
                self.emit_variable_decl(
                        identifier, self.name_manager[identifier], sym_kind)

            if sym_table:
                self.emit('')

        self.emit_trace('================================================')
        self.emit_trace('enter %s' % function_name)

        if state_id is not None:
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

    def emit_if_begin(self, expr):
        FortranIfEmitter(
                self.emitter,
                self.expr(expr),
                self).__enter__()

    def emit_if_end(self,):
        self.emitter.__exit__(None, None, None)

    def emit_else_begin(self):
        self.emitter.emit_else()

    def emit_assign_expr(self, assignee_sym, assignee_subscript, expr):
        from dagrt.vm.codegen.data import ODEComponent, Array

        assignee_fortran_name = self.name_manager[assignee_sym]

        sym_kind = self.sym_kind_table.get(
                self.current_function, assignee_sym)

        if assignee_subscript and not isinstance(sym_kind, Array):
            raise TypeError("only arrays support subscripted assignment")
            return

        if not isinstance(sym_kind, ODEComponent):
            self.emit_assign_expr_inner(
                    assignee_fortran_name, assignee_subscript, expr, sym_kind)
            return

        if assignee_subscript:
            raise ValueError("ODE components do not support subscripting")

        if isinstance(expr, Variable):
            self.emit_ode_component_move(
                    assignee_sym, assignee_fortran_name, sym_kind, expr)
            return

        from pymbolic import var
        from pymbolic.mapper.dependency import DependencyMapper

        # We can't tolerate reading a variable that we're just assigning,
        # as we need to make new storage for the assignee before the
        # read starts.
        assert var(assignee_sym) not in DependencyMapper()(expr)

        self.emit_allocation_check(assignee_sym, sym_kind)
        self.emit_assign_expr_inner(
                assignee_fortran_name, assignee_subscript, expr, sym_kind)

    def emit_inst_AssignExpression(self, inst):
        start_em = self.emitter

        for ident, start, stop in inst.loops:
            em = FortranDoEmitter(
                    self.emitter,
                    self.name_manager[ident],
                    "int(%s), int(%s)" % (self.expr(start), self.expr(stop-1)),
                    code_generator=self)
            em.__enter__()

        self.emit_assign_expr(
                inst.assignee, inst.assignee_subscript, inst.expression)

        for ident, start, stop in inst.loops[::-1]:
            self.emitter.__exit__(None, None, None)

        assert start_em is self.emitter

    def emit_inst_AssignFunctionCall(self, inst):
        self.emit_trace("func call {results} = {expr}..."
                .format(
                    results=", ".join(inst.assignees),
                    expr=str(inst.as_expression())[:50]))

        function = self.function_registry[inst.function_id]
        codegen = function.get_codegen(self.language)

        arg_strs_dict = {}
        arg_kinds_dict = {}
        for i, arg in enumerate(inst.parameters):
            arg_strs_dict[i] = self.expr(arg)
            assert isinstance(arg, Variable)

            # FIXME: This can fail for args of state update notification,
            # hence the try/catch.
            try:
                arg_kinds_dict[i] = self.sym_kind_table.get(
                        self.current_function, arg.name)
            except KeyError:
                arg_kinds_dict[i] = None

        for arg_name, arg in inst.kw_parameters.items():
            arg_strs_dict[arg_name] = self.expr(arg)
            assert isinstance(arg, Variable)

            # FIXME: This can fail for args of state update notification,
            # hence the try/catch.
            try:
                arg_kinds_dict[arg_name] = self.sym_kind_table.get(
                        self.current_function, arg.name)
            except KeyError:
                pass

        from pymbolic.mapper.dependency import DependencyMapper
        from pymbolic import var
        from dagrt.vm.codegen.data import ODEComponent

        for assignee_sym in inst.assignees:
            sym_kind = self.sym_kind_table.get(
                    self.current_function, assignee_sym)
            if isinstance(sym_kind, ODEComponent):
                self.emit_allocation_check(assignee_sym, sym_kind)

            assert var(assignee_sym) not in DependencyMapper()(
                    inst.as_expression())

        assignee_fortran_names = tuple(
                self.name_manager[assignee_sym] for a in inst.assignees)

        codegen(
                results=assignee_fortran_names,
                function=function,
                arg_strings_dict=arg_strs_dict,
                arg_kinds_dict=arg_kinds_dict,
                code_generator=self)

    def emit_return(self):
        self.emit('999 continue ! exit label')

        # {{{ emit variable deinit

        sym_table = self.sym_kind_table.per_function_table.get(
                self.current_function, {})

        for identifier, sym_kind in six.iteritems(sym_table):
            self.emit_variable_deinit(identifier, sym_kind)

        # }}}

        self.emit_trace('leave %s' % self.current_function)

        self.emit('return')

    def emit_inst_YieldState(self, inst):
        self.emit_assign_expr(
                '<ret_time_id>'+inst.component_id,
                (),
                Variable("leap_time_"+str(inst.time_id)))
        self.emit_assign_expr(
                '<ret_time>'+inst.component_id,
                (),
                inst.time)

        from dagrt.vm.language import AssignFunctionCall
        from pymbolic import var

        if self.call_before_state_update:
            self.emit_inst_AssignFunctionCall(
                    AssignFunctionCall(
                        (),
                        self.call_before_state_update,
                        (var(self.component_name_to_component_sym(
                            inst.component_id)),)))

        self.emit_assign_expr(
                '<ret_state>'+inst.component_id,
                (),
                inst.expression)

        if self.call_after_state_update:
            self.emit_inst_AssignFunctionCall(
                    AssignFunctionCall(
                        (),
                        self.call_after_state_update,
                        (var(self.component_name_to_component_sym(
                            inst.component_id)),)))

    def emit_inst_Raise(self, inst):
        self.emit("write (leap_stderr,*) "
                "'{condition}: {message}'".format(
                    condition=inst.error_condition.__name__,
                    message=inst.error_message))
        self.emit("stop")

    def emit_inst_FailStep(self, inst):
        self.emit("goto 999")

    def emit_inst_StateTransition(self, inst):
        self.emit(
                'leap_state%leap_next_state = '
                + self.state_name_to_state_sym(inst.next_state))

    # }}}

# }}}


# {{{ built-in functions

class TypeVisitorWithResult(TypeVisitor):
    def __init__(self, code_generator, result_expr):
        super(TypeVisitorWithResult, self).__init__(code_generator)
        self.result_expr = result_expr


class Norm2Computer(TypeVisitorWithResult):
    def visit_BuiltinType(self, fortran_type, fortran_expr, index_expr_map):
        self.code_generator.emit(
                "{result} = {result} + abs({expr})**2"
                .format(
                    result=self.result_expr,
                    expr=fortran_expr))


def codegen_builtin_norm_2(results, function, arg_strings_dict, arg_kinds_dict,
        code_generator):
    result, = results

    from dagrt.vm.codegen.data import Scalar, ODEComponent, Array
    x_kind = arg_kinds_dict[0]
    if isinstance(x_kind, Scalar):
        if x_kind.is_real_valued:
            ftype = BuiltinType("real*8")
        else:
            ftype = BuiltinType("complex*16")
    elif isinstance(x_kind, ODEComponent):
        ftype = code_generator.ode_component_type_map[x_kind.component_id]

    elif isinstance(x_kind, Array):
        code_generator.emit("{result} = norm2({arg})".format(
            result=result, arg=arg_strings_dict[0]))
        return

    else:
        raise TypeError("unsupported kind for norm_2 argument: %s" % x_kind)

    code_generator.emit("{result} = 0".format(result=result))
    code_generator.emit("")

    Norm2Computer(code_generator, result)(ftype, arg_strings_dict[0], {})

    code_generator.emit("")
    code_generator.emit("{result} = sqrt({result})".format(result=result))
    code_generator.emit("")


class LenComputer(TypeVisitorWithResult):
    # FIXME: This could be made *way* more efficient by handling
    # arrays of built-in types directly.

    def visit_BuiltinType(self, fortran_type, fortran_expr, index_expr_map):
        self.code_generator.emit(
                "{result} = {result} + 1"
                .format(
                    result=self.result_expr,
                    expr=fortran_expr))


def codegen_builtin_len(results, function, arg_strings_dict, arg_kinds_dict,
        code_generator):
    result, = results

    from dagrt.vm.codegen.data import Scalar, Array, ODEComponent
    x_kind = arg_kinds_dict[0]
    if isinstance(x_kind, Scalar):
        if x_kind.is_real_valued:
            ftype = BuiltinType("real*8")
        else:
            ftype = BuiltinType("complex*16")
    elif isinstance(x_kind, ODEComponent):
        ftype = code_generator.ode_component_type_map[x_kind.component_id]
    elif isinstance(x_kind, Array):
        code_generator.emit("{result} = size({arg})".format(
            result=result,
            arg=arg_strings_dict[0]))
        return
    else:
        raise TypeError("unsupported kind for norm_2 argument: %s" % x_kind)

    code_generator.emit("{result} = 0".format(result=result))
    code_generator.emit("")

    LenComputer(code_generator, result)(ftype, arg_strings_dict[0], {})
    code_generator.emit("")


class IsNaNComputer(TypeVisitorWithResult):
    def visit_BuiltinType(self, fortran_type, fortran_expr, index_expr_map):
        self.code_generator.emit(
                "{result} = {result} .or. isnan({expr})"
                .format(
                    result=self.result_expr,
                    expr=fortran_expr))


def codegen_builtin_isnan(results, function, arg_strings_dict, arg_kinds_dict,
        code_generator):
    result, = results

    from dagrt.vm.codegen.data import Scalar, ODEComponent
    x_kind = arg_kinds_dict[0]
    if isinstance(x_kind, Scalar):
        if x_kind.is_real_valued:
            ftype = BuiltinType("real*8")
        else:
            ftype = BuiltinType("complex*16")
    elif isinstance(x_kind, ODEComponent):
        ftype = code_generator.ode_component_type_map[x_kind.component_id]
    else:
        raise TypeError("unsupported kind for norm_2 argument: %s" % x_kind)

    code_generator.emit("{result} = .false.".format(result=result))
    code_generator.emit("")

    IsNaNComputer(code_generator, result)(ftype, arg_strings_dict[0], {})
    code_generator.emit("")


builtin_array = CallCode("""
        if (int(${n}).ne.${n}) then
            write(leap_stderr,*) 'argument to array() is not an integer'
            stop
        endif

        if (allocated(${result})) then
            deallocate(${result})
        endif

        allocate(${result}(0:int(${n})-1))
        """)


UTIL_MACROS = """
    <%def name="write_matrix(mat_array, rows_var)" >
        <%
        i = declare_new("integer", "i")
        %>
        do ${i} = 0, int(${rows_var})-1
            write(*,*) ${mat_array}(${i}::${rows_var})
        end do
    </%def>

    <%def name="check_matrix(mat_array, cols_var, rows_var, func_name)" >
        if (int(${cols_var}).ne.${cols_var}) then
            write(leap_stderr,*) &
                'argument ' // &
                '${cols_var} ' // &
                'to ${func_name}' // &
                'is not an integer'
            stop
        endif

        ${rows_var} = size(${mat_array}) / int(${cols_var})

        if (${rows_var} * int(${cols_var}) .ne. size(${mat_array})) then
            write(leap_stderr,*) &
                'size of argument ' // &
                '${mat_array}' // &
                'to ${func_name} ' // &
                'not divisible by ' // &
                '${cols_var}'
            stop
        endif
    </%def>

    <%
    def get_lapack_letter(kind):
        if kind.is_real_valued:
            if real_scalar_kind == "4":
                return "s"
            elif real_scalar_kind == "8":
                return "d"
            else:
                raise TypeError("unrecognized real kind %s" % real_scalar_kind)
        else:
            if complex_scalar_kind == "8":
                return "c"
            elif complex_scalar_kind == "16":
                return "z"
            else:
                raise TypeError("unrecognized complex kind %s"
                    % complex_scalar_kind)

    def kind_to_fortran(kind):
        if kind.is_real_valued:
            return "real (kind=%s)" % real_scalar_kind
        else:
            return "compelx (kind=%s)" % complex_scalar_kind
    %>

    """

builtin_matmul = CallCode(UTIL_MACROS + """
        <%
        a_rows = declare_new("integer", "a_rows")
        b_rows = declare_new("integer", "b_rows")
        res_size = declare_new("integer", "res_size")
        %>

        ${check_matrix(a, a_cols, a_rows, "matmul")}
        ${check_matrix(b, b_cols, b_rows, "matmul")}

        ${a_rows} = size(${a}) / int(${a_cols})
        ${b_rows} = size(${b}) / int(${b_cols})

        ${res_size} = ${a_rows} * int(${b_cols})

        if (allocated(${result})) then
            deallocate(${result})
        endif

        allocate(${result}(0:${res_size}-1))

        ${result} = reshape( &
                matmul( &
                    reshape(${a}, (/${a_rows}, int(${a_cols})/)), &
                    reshape(${b}, (/${b_rows}, int(${b_cols})/))), &
                (/${res_size}/))
        """)

builtin_linear_solve = CallCode(UTIL_MACROS + """
        <%
        a_rows = declare_new("integer", "a_rows")
        b_rows = declare_new("integer", "b_rows")
        res_size = declare_new("integer", "res_size")

        %>

        ${check_matrix(a, a_cols, a_rows, "linear_solve")}
        ${check_matrix(b, b_cols, b_rows, "linear_solve")}

        if (int(${a_rows}).ne.int(${b_rows})) then
            write(leap_stderr,*) 'inconsistent matrix sizes in linear_solve'
            stop
        endif
        if (int(${a_rows}).ne.int(${a_cols})) then
            write(leap_stderr,*) 'non-square matrix sizes in linear_solve'
            stop
        endif

        ${res_size} = int(${b_rows}) * int(${b_cols})

        <%
        if a_kind != b_kind:
            raise TypeError("linear_solve requires both arguments "
                "to have same kind")

        ltr = get_lapack_letter(a_kind)

        lu_temp = declare_new(
                kind_to_fortran(a_kind)+", dimension(:), allocatable"
                , "lu_temp")
        ipiv = declare_new("integer, dimension(:), allocatable", "ipiv")
        info = declare_new("integer", "info")
        %>

        allocate(${lu_temp}(0:size(${a})-1))
        allocate(${ipiv}(${a_rows}))

        ${lu_temp} = ${a}

        if (allocated(${result})) then
            deallocate(${result})
        endif

        allocate(${result}(0:${res_size}-1))
        ${result} = ${b}

        call ${ltr}gesv(int(${a_rows}), int(${b_cols}), &
            ${lu_temp}, int(${a_rows}), ${ipiv}, &
            ${result}, int(${b_rows}), ${info})

        if (${info}.ne.0) then
            write(leap_stderr,*) &
                'gesv on ${a} failed with info=', ${info}
            stop
        endif

        deallocate(${lu_temp})
        deallocate(${ipiv})

        """)

builtin_print = CallCode(UTIL_MACROS + """
        write(*,*) ${arg}
        """)

# }}}


# vim: foldmethod=marker
