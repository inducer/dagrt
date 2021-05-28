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

import sys
from functools import partial
import re  # noqa

from dagrt.codegen.expressions import FortranExpressionMapper
from dagrt.codegen.codegen_base import StructuredCodeGenerator
from dagrt.utils import is_state_variable
from dagrt.data import UserType
from pytools.py_codegen import (
        # It's the same code. So sue me.
        PythonCodeGenerator as FortranEmitterBase)
from pymbolic.primitives import (Call, CallWithKwargs, Variable,
        Subscript, Lookup)
from pymbolic.mapper import IdentityMapper
from dagrt.codegen.utils import (wrap_line_base, KeyToUniqueNameMap,
        make_identifier_from_name)


__doc__ = """
Types
-----
.. autoclass: TypeBase
.. autoclass: BuiltinType
.. autoclass: ArrayType
.. autoclass: PointerType
.. autoclass: StructureType

Code Generator
--------------

.. autoclass: CodeGenerator
"""


def pad_fortran(line, width):
    line += " " * (width - 1 - len(line))
    line += "&"
    return line


wrap_line = partial(wrap_line_base, pad_func=pad_fortran)


# {{{ name manager

class FortranNameManager:
    """Maps names that appear in intermediate code to Fortran identifiers.
    """

    def __init__(self):
        from pytools import UniqueNameGenerator
        self.name_generator = UniqueNameGenerator()
        self.local_map = KeyToUniqueNameMap(name_generator=self.name_generator)
        self.global_map = KeyToUniqueNameMap(start={
                "<t>": "dagrt_t", "<dt>": "dagrt_dt"},
                name_generator=self.name_generator)
        self.function_map = KeyToUniqueNameMap(name_generator=self.name_generator)

    def name_global(self, var):
        """Return the identifier for a global variable."""
        return self.global_map.get_or_make_name_for_key(var)

    def name_local(self, var, prefix=None):
        """Return the identifier for a local variable."""
        if prefix is None:
            if not var.startswith("dagrt_"):
                prefix = "lploc_"

        return self.local_map.get_or_make_name_for_key(var, prefix=prefix)

    def name_function(self, var):
        """Return the identifier for a function."""
        return self.function_map.get_or_make_name_for_key(var)

    def make_unique_fortran_name(self, prefix):
        return self.local_map.get_mapped_identifier_without_key("drtf_"+prefix)

    def is_known_fortran_name(self, name):
        return self.name_generator.is_name_conflicting(name)

    def __getitem__(self, name):
        """Provide an interface to the expression mapper to look up
        the name of a local or global variable.
        """
        if is_state_variable(name):
            return "dagrt_state%"+self.name_global(name)
        else:
            return self.name_local(name)

    def name_refcount(self, name, qualified_with_state=True):
        if is_state_variable(name):
            if qualified_with_state:
                return "dagrt_state%dagrt_refcnt_"+self.name_global(name)
            else:
                return "dagrt_refcnt_"+self.name_global(name)
        else:
            return self.name_local("dagrt_refcnt_"+name)

# }}}


# {{{ custom emitters

class FortranEmitter(FortranEmitterBase):
    def incorporate(self, sub_generator):
        for line in sub_generator.code:
            self(line)


class FortranBlockEmitter(FortranEmitter):
    def __init__(self, what, code_generator=None):
        super().__init__()
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
        self(f"end {self.what}")
        self("")


class FortranModuleEmitter(FortranBlockEmitter):
    def __init__(self, module_name):
        super().__init__("module")
        self.module_name = module_name
        self(f"module {module_name}")


class FortranSubblockEmitter(FortranBlockEmitter):
    def __init__(self, parent_emitter, what, code_generator=None):
        super().__init__(what, code_generator)
        self.parent_emitter = parent_emitter

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(
                exc_type, exc_val, exc_tb)

        self.parent_emitter.incorporate(self)


class FortranIfEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, expr, code_generator=None):
        super().__init__(
                parent_emitter, "if", code_generator)
        self(f"if ({expr}) then")

    def emit_else(self):
        self.dedent()
        self("else")
        self.indent()

    def emit_else_if(self, expr):
        self.dedent()
        self(f"else if ({expr}) then")
        self.indent()


class FortranDoEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, loop_var, bounds, code_generator=None,
            parallel_do_preamble=None):
        super().__init__(
                parent_emitter, "do", code_generator)
        if parallel_do_preamble:
            self(parallel_do_preamble)

        self("do {loop_var} = {bounds}".format(
            loop_var=loop_var, bounds=bounds))


class FortranSubroutineEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, name, args, code_generator=None):
        super().__init__(
                parent_emitter, "subroutine", code_generator)
        self.name = name

        self("subroutine {}({})".format(name, ", ".join(args)))


class FortranTypeEmitter(FortranSubblockEmitter):
    def __init__(self, parent_emitter, type_name, code_generator=None):
        super().__init__(
                parent_emitter, "type", code_generator)
        self(f"type {type_name}")

# }}}


# {{{ code generation for function calls

class CallCode:
    """Encapsulation for a Fortran code template embodying a dagrt-level function call.
    """

    def __init__(self, template, extra_args=None):
        """
        :arg extra_args: a dictionary of names that should be made available
            in template expansion.
        """

        from mako.template import Template

        self.template = Template(template, strict_undefined=True)

        self.extra_args = extra_args

    def __call__(self, results, function, args, arg_kinds,
            code_generator):
        from dagrt.codegen.utils import (
                remove_common_indentation,
                remove_redundant_blank_lines)

        def add_declaration(decl):
            code_generator.declaration_emitter(decl)

        def declare_new(decl_without_name, prefix):
            new_name = code_generator.name_manager.make_unique_fortran_name(prefix)
            code_generator.declaration_emitter(decl_without_name + " :: " + new_name)
            return new_name

        import dagrt.data as kinds

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

        for line in lines:
            code_generator.emit(line)

# }}}


# {{{ expression modifiers

class UserTypeReferenceTransformer(IdentityMapper):
    def __init__(self, code_generator):
        self.code_generator = code_generator

    def find_sym_kind(self, expr):
        if isinstance(expr, (Subscript, Lookup)):
            return self.find_sym_kind(expr.aggregate)
        elif isinstance(expr, Variable):
            return self.code_generator.sym_kind_table.get(
                    self.code_generator.current_function, expr.name)
        else:
            raise TypeError("unsupported object")

    def transform(self, expr):
        raise NotImplementedError

    def map_variable(self, expr):
        if isinstance(self.find_sym_kind(expr), UserType):
            return self.transform(expr)
        else:
            return expr

    map_lookup = map_variable
    map_subscript = map_variable


class StructureLookupAppender(UserTypeReferenceTransformer):
    def __init__(self, code_generator, component):
        super().__init__(code_generator)
        self.component = component

    def transform(self, expr):
        return expr.attr(self.component)


class ArraySubscriptAppender(UserTypeReferenceTransformer):
    def __init__(self, code_generator, subscript):
        super().__init__(code_generator)
        self.subscript = subscript

    def transform(self, expr):
        return expr[self.subscript]

# }}}


# {{{ fortran "vector-ish" types

class TypeBase:
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
    """
    .. attribute:: pointee_type
    .. attribute:: is_synthetic

        A :class:`bool` flag indicating whether this pointer declaration
        is genuinely part of the user type (False) or 'synthetically' inserted
        by the code generator (True).
    """

    def __init__(self, pointee_type, is_synthetic=False):
        self.pointee_type = pointee_type
        self.is_synthetic = is_synthetic

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
    for name, expr in index_expr_map.items():
        s, _ = re.subn(r"\b" + name + r"\b", expr, s)
    return s


class _ArrayLoopManager:
    def __init__(self, array_type, code_generator):
        self.array_type = array_type
        self.code_generator = code_generator

        self.f_index_names = [
            code_generator.name_manager.make_unique_fortran_name(iname)
            for iname in array_type.index_vars]

        self.f_dim_names = [
            code_generator.name_manager.make_unique_fortran_name(iname[-6:])
            for iname in array_type.dimension]

    def enter(self, index_expr_map, allow_parallel_do):
        atype = self.array_type

        cg = self.code_generator

        self.emitters = []
        for iloop, (dim, index_name, dim_name) in enumerate(
                reversed(list(zip(atype.dimension, self.f_index_names,
                    self.f_dim_names)))):
            cg.declaration_emitter("integer %s" % index_name)
            cg.declaration_emitter("integer %s" % dim_name)

            pdp = None
            if allow_parallel_do and iloop + 1 == len(atype.dimension):
                pdp = cg.parallel_do_preamble

            start, stop = atype.parse_dimension(dim)

            self.code_generator.emit(
                    "{name} = {expr}"
                    .format(
                        name=dim_name,
                        expr=_replace_indices(index_expr_map, stop)))

            em = FortranDoEmitter(
                    cg.emitter, index_name,
                    "{}, {}".format(
                        _replace_indices(index_expr_map, start),
                        _replace_indices(index_expr_map, dim_name)),
                    cg,
                    parallel_do_preamble=pdp)
            self.emitters.append(em)
            em.__enter__()

    def update_index_expr_map(self, index_expr_map):
        index_expr_map.update(zip(self.array_type.index_vars, self.f_index_names))

    def get_loop_subscript(self):
        from pymbolic import var
        return tuple(var("<target>"+v) for v in self.f_index_names)

    def leave(self):
        while self.emitters:
            em = self.emitters.pop()
            em.__exit__(None, None, None)

# }}}


class TypeVisitor:
    recurse_only_if_allocatable = False

    def rec(self, fortran_type, *args, **kwargs):
        return getattr(self, "visit_"+type(fortran_type).__name__)(
                fortran_type, *args, **kwargs)

    __call__ = rec


class DeclarationGenerator(TypeVisitor):
    def __init__(self, use_deferred_shape):
        self.use_deferred_shape = use_deferred_shape

    def visit_BuiltinType(self, fortran_type):
        return (fortran_type.type_name,)

    def visit_ArrayType(self, fortran_type):
        if self.use_deferred_shape:
            dim_spec = ", ".join(":" for s in fortran_type.dimension)
        else:
            dim_spec = ", ".join(fortran_type.dimension)

        return self.rec(fortran_type.element_type) + (
                "dimension(%s)" % dim_spec,)

    def visit_PointerType(self, fortran_type):
        return self.rec(fortran_type.pointee_type) + ("pointer",)

    def visit_StructureType(self, fortran_type):
        return ("type(%s)" % fortran_type.type_name,)


class CodeGeneratingTypeVisitor(TypeVisitor):
    def __init__(self, code_generator):
        self.code_generator = code_generator

    def visit_BuiltinType(self, fortran_type, fortran_expr_str, index_expr_map,
            *args, **kwargs):
        pass

    def visit_ArrayType(self, fortran_type, fortran_expr_str, index_expr_map,
            *args, **kwargs):
        if (self.recurse_only_if_allocatable
                and not fortran_type.element_type.is_allocatable()):
            return

        alm = _ArrayLoopManager(fortran_type, self.code_generator)
        alm.enter(index_expr_map, allow_parallel_do=False)

        index_expr_map = index_expr_map.copy()
        alm.update_index_expr_map(index_expr_map)

        self.rec(fortran_type.element_type,
                "{}({})".format(fortran_expr_str, ", ".join(alm.f_index_names)),
                index_expr_map, *args, **kwargs)

        alm.leave()

    def visit_PointerType(self, fortran_type, fortran_expr_str, index_expr_map,
            *args, **kwargs):
        if (self.recurse_only_if_allocatable
                and not fortran_type.pointee_type.is_allocatable()):
            return

        self.rec(fortran_type.pointee_type, fortran_expr_str, index_expr_map,
                *args, **kwargs)

    def visit_StructureType(self, fortran_type, fortran_expr_str, index_expr_map,
            *args, **kwargs):
        for member_name, member_type in fortran_type.members:
            if (self.recurse_only_if_allocatable
                    and not member_type.is_allocatable()):
                continue

            self.rec(member_type,
                    fortran_expr_str+"%"+member_name,
                    index_expr_map,
                    *args, **kwargs)


class PointerAliasCreatingArraySubscriptAppender(ArraySubscriptAppender):
    """Used to hoist array base subexpressions out of assignments."""

    def __init__(self, code_generator, subscript, fortran_type):
        super().__init__(
                code_generator, subscript)
        self.expr_to_alias = {}
        self.fortran_type = fortran_type

    def transform(self, expr):
        try:
            expr_fortran_name = self.expr_to_alias[expr]
        except KeyError:
            expr_fortran_name = (
                    self.code_generator.name_manager.make_unique_fortran_name(
                        "hoisted"))

            dg = DeclarationGenerator(use_deferred_shape=True)
            self.code_generator.declaration_emitter(
                    ", ".join(dg(self.fortran_type) + ("pointer",))
                    + " :: " + expr_fortran_name)

            self.code_generator.emit(
                    "{name} => {expr}"
                    .format(
                        name=expr_fortran_name,
                        expr=self.code_generator.expr(expr)))

            self.expr_to_alias[expr] = expr_fortran_name

        from pymbolic import var
        return var("<target>"+expr_fortran_name)[self.subscript]


class AssignmentEmitter(CodeGeneratingTypeVisitor):
    def visit_BuiltinType(self, fortran_type, fortran_expr_str, index_expr_map,
            rhs_expr, is_rhs_target):
        self.code_generator.emit(
                "{name} = {expr}"
                .format(
                    name=fortran_expr_str,
                    expr=self.code_generator.expr(rhs_expr)))

    def visit_ArrayType(self, fortran_type, fortran_expr_str, index_expr_map,
            rhs_expr, is_rhs_target):
        el_is_primitive = isinstance(fortran_type.element_type, BuiltinType)

        cg = self.code_generator

        alm = _ArrayLoopManager(fortran_type, cg)

        if el_is_primitive and is_rhs_target:
            lhs_fortran_name = cg.name_manager.make_unique_fortran_name("hoisted")

            dg = DeclarationGenerator(use_deferred_shape=True)
            cg.declaration_emitter(
                    ", ".join(dg(fortran_type) + ("pointer",))
                    + " :: " + lhs_fortran_name)

            cg.emit(
                    "{name} => {expr}"
                    .format(
                        name=lhs_fortran_name,
                        expr=fortran_expr_str))

            transformer = PointerAliasCreatingArraySubscriptAppender(
                    self.code_generator, alm.get_loop_subscript(),
                    fortran_type=fortran_type)

            # This generates a number of variable declarations
            # and assignments, so it must happen before we
            # enter the loop.
            rhs_expr = transformer(rhs_expr)

        else:
            lhs_fortran_name = fortran_expr_str
            transformer = ArraySubscriptAppender(
                    self.code_generator, alm.get_loop_subscript())

            rhs_expr = transformer(rhs_expr)

        alm.enter(index_expr_map,
                allow_parallel_do=el_is_primitive)
        index_expr_map = index_expr_map.copy()
        alm.update_index_expr_map(index_expr_map)

        self.rec(fortran_type.element_type,
                "{}({})".format(lhs_fortran_name, ", ".join(alm.f_index_names)),
                index_expr_map, rhs_expr, is_rhs_target=is_rhs_target)

        alm.leave()

    def visit_PointerType(self, fortran_type, fortran_expr_str, index_expr_map,
            rhs_expr, is_rhs_target):
        if (self.recurse_only_if_allocatable
                and not fortran_type.pointee_type.is_allocatable()):
            return

        self.rec(fortran_type.pointee_type, fortran_expr_str, index_expr_map,
                rhs_expr, is_rhs_target=True and not fortran_type.is_synthetic)

    def visit_StructureType(self, fortran_type, fortran_expr_str, index_expr_map,
            rhs_expr, is_rhs_target):
        for member_name, member_type in fortran_type.members:
            sla = StructureLookupAppender(self.code_generator, member_name)

            self.rec(member_type,
                    fortran_expr_str+"%"+member_name,
                    index_expr_map,
                    sla(rhs_expr),
                    is_rhs_target=is_rhs_target)


class AllocationEmitter(CodeGeneratingTypeVisitor):
    recurse_only_if_allocatable = True

    def visit_PointerType(self, fortran_type, fortran_expr_str, index_expr_map):
        pointee_type = fortran_type.pointee_type
        code_generator = self.code_generator

        dimension = ""
        if isinstance(pointee_type, ArrayType):
            if pointee_type.dimension:
                dimension = "(%s)" % ", ".join(
                        str(dim_axis) for dim_axis in pointee_type.dimension)

        code_generator.emit_traceable(
                "allocate({name}{dimension}, stat=dagrt_ierr)".format(
                    name=fortran_expr_str,
                    dimension=_replace_indices(index_expr_map, dimension)))
        with FortranIfEmitter(
                code_generator.emitter, "dagrt_ierr.ne.0", code_generator):
            code_generator.emit(
                    "write(dagrt_stderr,*) 'failed to allocate {name}'".format(
                        name=fortran_expr_str))
            code_generator.emit("stop")

        if pointee_type.is_allocatable():
            self.rec(pointee_type, fortran_expr_str, index_expr_map)


class DeallocationEmitter(CodeGeneratingTypeVisitor):
    recurse_only_if_allocatable = True

    def __init__(self, code_generator, deinitializer):
        super().__init__(code_generator)
        self.deinitializer = deinitializer

    def visit_PointerType(self, fortran_type, fortran_expr_str, index_expr_map):
        pointee_type = fortran_type.pointee_type
        code_generator = self.code_generator

        if pointee_type.is_allocatable():
            self.rec(pointee_type, fortran_expr_str, index_expr_map)
        self.deinitializer(pointee_type, fortran_expr_str, index_expr_map)

        code_generator.emit_traceable(f"deallocate({fortran_expr_str})")
        code_generator.emit_traceable("nullify(%s)" % fortran_expr_str)


class InitializationEmitter(CodeGeneratingTypeVisitor):
    recurse_only_if_allocatable = True

    def visit_PointerType(self, fortran_type, fortran_expr_str, index_expr_map):
        self.code_generator.emit_traceable("nullify(%s)" % fortran_expr_str)

# }}}


# {{{ code generator

class CodeGenerator(StructuredCodeGenerator):
    """
    Generates a Fortran module of name *module_name*, which defines a type
    *dagrt_state_type* to hold the current state of the time integrator
    along with several functions::

        initialize(EXTRA_ARGUMENTS, dagrt_state, ...)
        run(EXTRA_ARGUMENTS, dagrt_state)
        shutdown(EXTRA_ARGUMENTS, dagrt_state)
        print_profile(dagrt_state)

    *dagrt_state* is of type *dagrt_state_type`, and *EXTRA_ARGUMENTS* above matches
    *extra_arguments* as passed to the constructor.

    The ``...`` arguments to ``initialize`` are optional and must be passed by
    keyword.  The following keywords arguments are available:

    * *dagrt_dt*: The initial time step size
    * *dagrt_t*: The initial time
    * *state_STATE*: The initial value for the :mod:`dagrt` variable ``<state>STATE``

    .. rubric:: Profiling information

    The following attributes are available and allowed for read access in
    *dagrt_state_type* while outside of *run*:

    * *dagrt_state_PHASE_count*
    * *dagrt_state_PHASE_failures*
    * *dagrt_state_PHASE_time*

    * *dagrt_func_FUNC_count*
    * *dagrt_func_FUNC_time*

    In all of the above, upper case denotes a "metavariable"--e.g. *PHASE* is
    the name of a phase, or *FUNC* is  the name of a function. The name of a
    function will typically be ``<func>something``, for which *FUNC* will be
    ``func_something``.  As a result, the profile field counting the number of
    invocations of the function ``<func>something`` will be named
    *dagrt_func_func_something*.
    """

    language = "fortran"

    # {{{ constructor

    def __init__(self, module_name,
            user_type_map,
            function_registry=None,
            module_preamble=None,
            real_scalar_kind="8",
            complex_scalar_kind="8",
            use_complex_scalars=True,
            call_before_state_update=None,
            call_after_state_update=None,
            extra_arguments=(),
            extra_argument_decl=None,

            parallel_do_preamble=None,

            emit_instrumentation=False,
            timing_function=None,

            trace=False):
        """
        :arg function_registry: An instance of
            :class:`dagrt.function_registry.FunctionRegistry`
        :arg module_preamble: A string to include at the beginning of the
            emitted module
        :arg user_type_map: a map from user type names
            to :class:`TypeBase` instances
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
        :arg parallel_do_preamble: *None* or a string to be inserted before
            each constant-trip-count simd'able ``do`` loop.
        :arg emit_instrumentation: True or False, whether to emit performance
            instrumentation.
        :arg timing_function: *None* or the name of a function that returns
            wall clock time as a number of seconds, as a ``real*8``.
            Required if *emit_instrumentation* is set to *True*.
        """
        if function_registry is None:
            from dagrt.function_registry import base_function_registry
            function_registry = base_function_registry

        for type_name, type_val in user_type_map.items():
            # Because if it's already a pointer, we have a hard time declaring
            # the input type of our memory management routines.

            if isinstance(type_val, PointerType):
                raise ValueError("type '%s': PointerType is not allowed as the "
                        "outermost type in user type mappings"
                        % type_name)

        self.module_name = module_name
        self.function_registry = function_registry
        self.user_type_map = user_type_map

        self.trace = trace

        from dagrt.codegen.utils import remove_common_indentation
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
            from dagrt.codegen.utils import remove_common_indentation
            extra_argument_decl = remove_common_indentation(
                extra_argument_decl)
        self.extra_argument_decl = extra_argument_decl

        self.parallel_do_preamble = parallel_do_preamble
        self.emit_instrumentation = emit_instrumentation
        self.timing_function = timing_function
        if emit_instrumentation and timing_function is None:
            raise ValueError("must supply timing_function if "
                    "emit_instrumentation is True")

        self.name_manager = FortranNameManager()
        self.expr_mapper = FortranExpressionMapper(
                self.name_manager)

        self.function_and_arg_kinds_to_fortran_name = {}

        # FIXME: Should make extra arguments known to
        # name manager

        self.module_emitter = FortranModuleEmitter(module_name)
        self.module_emitter.__enter__()

        self.emitters = [self.module_emitter]

        self.current_function = None
        self.used = False

    # }}}

    # {{{ utilities

    def get_called_function_names(self, dag):
        from dagrt.codegen.analysis import collect_function_names_from_dag
        result = collect_function_names_from_dag(dag)

        if self.call_before_state_update:
            result.add(self.call_before_state_update)

        if self.call_after_state_update:
            result.add(self.call_after_state_update)

        return sorted(result)

    def get_alloc_check_name(self, utype_id):
        return "dagrt_alloc_check_"+make_identifier_from_name(utype_id)

    def get_var_deinit_name(self, utype_id):
        return "dagrt_deinit_"+make_identifier_from_name(utype_id)

    @staticmethod
    def phase_name_to_phase_sym(phase_name):
        return "dagrt_phase_"+phase_name

    @staticmethod
    def component_name_to_component_sym(comp_name):
        return "dagrt_component_"+comp_name

    @property
    def emitter(self):
        return self.emitters[-1]

    def expr(self, expr):
        return self.expr_mapper(expr)

    def emit(self, line):
        self.emitter(line)

    def emit_traceable(self, line):
        self.emit_trace(line)
        self.emit(line)

    def emit_trace(self, line):
        if self.trace:
            self.emit("write(*,*) '%s'" % line)

    # }}}

    # {{{ main entrypoint

    def __call__(self, dag):
        if self.used:
            raise RuntimeError("fortran code generator may not be "
                    "used more than once")
        self.used = True

        from dagrt.codegen.analysis import verify_code
        verify_code(dag)

        # from dagrt.language import show_dependency_graph
        # show_dependency_graph(dag)

        # {{{ produce function name / function AST pairs

        from dagrt.codegen.dag_ast import (
                create_ast_from_phase, get_statements_in_ast)

        from collections import namedtuple
        NameASTPair = namedtuple("NameASTPair", "name, ast")  # noqa
        fdescrs = []

        def process_ast(ast, print_ast=False):
            from dagrt.codegen.transform import (
                    eliminate_self_dependencies,
                    isolate_function_arguments,
                    isolate_function_calls,
                    expand_IfThenElse)
            ast = eliminate_self_dependencies(ast)
            ast = isolate_function_arguments(ast)
            ast = isolate_function_calls(ast)
            ast = expand_IfThenElse(ast)

            if print_ast:
                print(ast)

            return ast

        for phase_name in sorted(dag.phases.keys()):
            ast = create_ast_from_phase(dag, phase_name)
            fdescrs.append(NameASTPair(phase_name, process_ast(ast)))

        # }}}

        from dagrt.data import SymbolKindFinder, Integer
        from dagrt.codegen.dag_ast import LoopVariableFinder

        self.sym_kind_table = SymbolKindFinder(self.function_registry)(
                [fd.name for fd in fdescrs],
                [get_statements_in_ast(fd.ast) for fd in fdescrs],
                forced_kinds=[
                    (fd.name, loop_var, Integer())
                    for fd in fdescrs
                    for loop_var in LoopVariableFinder()(fd.ast)])

        from dagrt.codegen.analysis import (
                collect_ode_component_names_from_dag,
                var_to_last_dependent_statement_mapping)

        component_ids = collect_ode_component_names_from_dag(dag)

        self.last_used_stmt_table = var_to_last_dependent_statement_mapping(
            [fd.name for fd in fdescrs],
            [get_statements_in_ast(fd.ast) for fd in fdescrs])

        if not component_ids <= set(self.user_type_map):
            raise RuntimeError("User type missing from user type map: %r"
                    % (component_ids - set(self.user_type_map)))

        from dagrt.data import Scalar, UserType
        for comp_id in component_ids:
            self.sym_kind_table.set(
                    None, "<ret_time_id>"+comp_id, Scalar(is_real_valued=True))
            self.sym_kind_table.set(
                    None, "<ret_time>"+comp_id, Scalar(is_real_valued=True))
            self.sym_kind_table.set(
                    None, "<ret_state>"+comp_id, UserType(comp_id))

        self.begin_emit(dag)
        for fdescr in sorted(fdescrs, key=lambda fdescr: fdescr.name):
            self.lower_function(fdescr.name, fdescr.ast)
        self.finish_emit(dag)

        del self.sym_kind_table

        code = self.get_code()

        new_lines = []
        for ln in code.split("\n"):
            if ln.lstrip().startswith("#"):
                hashmark_pos = ln.find("#")
                assert hashmark_pos >= 0
                ln = "#" + ln[:hashmark_pos] + ln[hashmark_pos+1:]

            new_lines.append(ln)

        return "\n".join(new_lines)

    # }}}

    # {{{ lower_function

    def lower_function(self, function_name, ast):
        self.current_function = function_name

        self.emit_def_begin(
                "dagrt_phase_func_" + function_name,
                self.extra_arguments + ("dagrt_state",),
                phase_id=function_name)
        self.declaration_emitter("type(dagrt_state_type), pointer :: dagrt_state")
        self.declaration_emitter("")

        # {{{ instrumentation

        if self.emit_instrumentation:
            self.emit(
                    "dagrt_state%dagrt_phase_{phase}_count "
                    "= dagrt_state%dagrt_phase_{phase}_count + 1"
                    .format(phase=function_name))

            timer_start_var = self.name_manager.make_unique_fortran_name(
                "timer_start")
            self.declaration_emitter("real*8 " + timer_start_var)

            self.emit(
                    "{timer_start_var} = {timing_function}()"
                    .format(
                        timer_start_var=timer_start_var,
                        timing_function=self.timing_function))

        # }}}

        self.lower_ast(ast)

        self.emit("999 continue ! exit label")

        # {{{ emit variable deinit

        sym_table = self.sym_kind_table.per_phase_table.get(
                self.current_function, {})

        for identifier, sym_kind in sorted(sym_table.items()):
            if (identifier, self.current_function) not in self.last_used_stmt_table:
                self.emit_variable_deinit(identifier, sym_kind)

        # }}}

        self.emit_trace("leave %s" % self.current_function)

        # {{{ instrumentation

        if self.emit_instrumentation:
            self.emit(
                    "dagrt_state%dagrt_phase_{phase}_time "
                    "= dagrt_state%dagrt_phase_{phase}_time "
                    "+ ({timing_function}() - {timer_start_var})"
                    .format(
                        phase=function_name,
                        timing_function=self.timing_function,
                        timer_start_var=timer_start_var,
                        ))

        # }}}

        self.emit_def_end(function_name)

        self.current_function = None

    # }}}

    # {{{ get_code

    def get_code(self):
        assert not self.module_emitter.preamble

        indent_spaces = 1
        indentation = indent_spaces*" "

        wrapped_lines = []
        for line in self.module_emitter.code:
            line_leading_spaces = (len(line) - len(line.lstrip(" ")))
            level = line_leading_spaces // indent_spaces
            line_ind = level*indentation
            if line[line_leading_spaces:].startswith("!"):
                wrapped_lines.append(line)
            else:
                for wrapped_line in wrap_line(
                        line[line_leading_spaces:],
                        level, indentation=indentation):
                    wrapped_lines.append(line_ind+wrapped_line)

        return "\n".join(wrapped_lines)

    # }}}

    # {{{ begin/finish_emit

    def begin_emit(self, dag):
        if self.module_preamble:
            for line in self.module_preamble:
                self.emit(line)
            self.emit("")

        from dagrt.codegen.analysis import collect_time_ids_from_dag
        for i, time_id in enumerate(sorted(collect_time_ids_from_dag(dag))):
            self.emit(f"integer dagrt_time_{time_id}")
            self.emit("parameter (dagrt_time_{time_id} = {i})".format(
                time_id=time_id, i=i))
        self.emit("")

        # {{{ phase name constants

        for i, phase in enumerate(sorted(dag.phases)):
            phase_sym_name = self.phase_name_to_phase_sym(phase)
            self.emit("integer {phase_sym_name}".format(
                phase_sym_name=phase_sym_name))
            self.emit("parameter ({phase_sym_name} = {i})".format(
                phase_sym_name=phase_sym_name, i=i))

        self.emit("")

        # }}}

        # {{{ component name constants

        from dagrt.codegen.analysis import collect_ode_component_names_from_dag
        component_ids = collect_ode_component_names_from_dag(dag)

        for i, comp_id in enumerate(sorted(component_ids)):
            comp_sym_name = self.component_name_to_component_sym(comp_id)
            self.emit("integer {comp_sym_name}".format(
                comp_sym_name=comp_sym_name))
            self.emit("parameter ({comp_sym_name} = {i})".format(
                comp_sym_name=comp_sym_name,
                i=i))

        self.emit("")

        # }}}

        # {{{ state type

        with FortranTypeEmitter(
                self.emitter,
                "dagrt_state_type",
                self) as emit:
            emit("integer dagrt_next_phase")
            emit("")

            for identifier, sym_kind in sorted(
                    self.sym_kind_table.global_table.items()):
                self.emit_variable_decl(
                        self.name_manager.name_global(identifier),
                        sym_kind=sym_kind,
                        refcount_name=self.name_manager.name_refcount(
                            identifier, qualified_with_state=False))

            # {{{ instrumentation

            if self.emit_instrumentation:
                emit("")
                emit("! {{{ instrumentation")
                emit("")

                for phase_name in sorted(dag.phases):
                    emit("integer dagrt_phase_%s_count" % phase_name)
                    emit("integer dagrt_phase_%s_failures" % phase_name)
                    emit("real*8 dagrt_phase_%s_time" % phase_name)

                emit("")

                for func_name in self.get_called_function_names(dag):
                    func_id = make_identifier_from_name(func_name)
                    emit("integer dagrt_func_%s_count" % func_id)
                    emit("real*8 dagrt_func_%s_time" % func_id)

                emit("")
                emit("! }}}")
                emit("")

            # }}}

        # }}}

        self.emit("integer dagrt_stderr")
        self.emit("parameter (dagrt_stderr=0)")
        self.emit("")
        self.emit("contains")
        self.emit("")

        # {{{ memory management routines

        from dagrt.data import collect_user_types
        user_types = collect_user_types(self.sym_kind_table)

        # {{{ allocation checks

        for utype_id in sorted(user_types):
            val_name = make_identifier_from_name(utype_id)
            function_name = self.get_alloc_check_name(utype_id)

            self.emit_def_begin(function_name,
                    self.extra_arguments + (val_name, "refcount"))

            sym_kind = UserType(utype_id)
            self.emit_variable_decl(val_name, sym_kind,
                    is_argument=True, emit=self.declaration_emitter,
                    other_specifiers=("pointer",))
            self.declaration_emitter("integer, pointer :: refcount")
            self.declaration_emitter("")

            with FortranIfEmitter(
                    self.emitter,
                    ".not.associated(%s)" % val_name,
                    self) as emit_if:

                ftype = self.get_fortran_type_for_user_type(utype_id)
                AllocationEmitter(self)(ftype, val_name, {})

                self.emit_allocate_refcount("refcount")

                # https://github.com/PyCQA/pylint/issues/3234
                emit_if.emit_else()  # pylint:disable=no-member

                # If the refcount is 1, then nobody else is referring to
                # the memory, and we might as well repurpose/overwrite it,
                # so there's nothing more to do in that case.

                with FortranIfEmitter(
                        self.emitter, "refcount.ne.1", self) as emit_if:

                    self.emit_traceable(
                            "refcount = refcount - 1")

                    # We get here if the refcount is not 1 initially, which
                    # means it's not zero here--someone else is still
                    # referring to the data. Let them have it, we'll make
                    # a new array.

                    self.emit("")

                    AllocationEmitter(self)(ftype, val_name, {})
                    self.emit_allocate_refcount("refcount")

            self.emit_def_end(function_name)

        # }}}

        # {{{ deinit

        for utype_id in sorted(user_types):
            val_name = make_identifier_from_name(utype_id)
            function_name = self.get_var_deinit_name(utype_id)

            self.emit_def_begin(function_name,
                    self.extra_arguments + (val_name, "refcount"))

            sym_kind = UserType(utype_id)
            self.emit_variable_decl(val_name, sym_kind,
                    is_argument=True, emit=self.declaration_emitter,
                    other_specifiers=("pointer",))
            self.declaration_emitter("integer, pointer :: refcount")
            self.declaration_emitter("")

            with FortranIfEmitter(
                    self.emitter, "associated(%s)" % val_name, self):
                with FortranIfEmitter(
                        self.emitter, "refcount.eq.1", self) \
                                as if_emit:
                    ftype = self.get_fortran_type_for_user_type(sym_kind.identifier)
                    DeallocationEmitter(self, InitializationEmitter(self))(
                            ftype, val_name, {})

                    self.emit_traceable("deallocate(refcount)")

                    # https://github.com/PyCQA/pylint/issues/3234
                    if_emit.emit_else()  # pylint:disable=no-member

                    InitializationEmitter(self)(ftype, val_name, {})
                    self.emit_traceable("refcount = refcount - 1")

            self.emit_def_end(function_name)

        # }}}

        # }}}

    def finish_emit(self, dag):
        for (function_id, arg_kinds), fortran_name in \
                self.function_and_arg_kinds_to_fortran_name.items():
            self.emit_dagrt_function(fortran_name, function_id, arg_kinds)

        self.emit_initialize(dag)
        self.emit_shutdown()
        self.emit_run_step(dag)

        self.emit_print_profile(dag)

        self.module_emitter.__exit__(None, None, None)

        self.emit("! vim:foldmethod=marker:filetype=fortran")

    # }}}

    # {{{ data management

    def get_fortran_type_for_user_type(self, type_identifier, is_argument=False):
        ftype = self.user_type_map[type_identifier]
        if not is_argument:
            ftype = PointerType(ftype, is_synthetic=True)

        return ftype

    def emit_variable_decl(self, fortran_name, sym_kind,
            is_argument=False, other_specifiers=(), emit=None,
            refcount_name=None):
        if emit is None:
            emit = self.emit

        from dagrt.data import Boolean, Scalar, Array, Integer

        type_specifiers = other_specifiers

        from dagrt.data import UserType
        if isinstance(sym_kind, Boolean):
            type_name = "logical"

        elif isinstance(sym_kind, Array):
            if sym_kind.is_real_valued or not self.use_complex_scalars:
                type_name = "real (kind=%s)" % self.real_scalar_kind
            else:
                type_name = "complex (kind=%s)" % self.complex_scalar_kind
            type_specifiers = type_specifiers + ("allocatable", "dimension(:)")

        elif isinstance(sym_kind, Scalar):
            if sym_kind.is_real_valued or not self.use_complex_scalars:
                type_name = "real (kind=%s)" % self.real_scalar_kind
            else:
                type_name = "complex (kind=%s)" % self.complex_scalar_kind

        elif isinstance(sym_kind, Integer):
            type_name = "integer"

        elif isinstance(sym_kind, UserType):
            ftype = self.get_fortran_type_for_user_type(sym_kind.identifier,
                    is_argument=is_argument)

            type_name = ftype.get_base_type()
            type_specifiers = (
                    type_specifiers
                    + ftype.get_type_specifiers(defer_dim=is_argument))

            if not is_argument:
                emit(
                        "integer, pointer ::  {refcount_name}".format(
                            refcount_name=refcount_name))

        else:
            raise ValueError("unknown variable kind: %s" % type(sym_kind).__name__)

        if type_specifiers:
            emit("{type_name}, {type_specifier_list} :: {id}".format(
                type_name=type_name,
                type_specifier_list=", ".join(type_specifiers),
                id=fortran_name))
        else:
            emit("{type_name} {id}".format(
                type_name=type_name,
                id=fortran_name))

    def emit_variable_init(self, name, sym_kind):
        from dagrt.data import UserType
        if isinstance(sym_kind, UserType):
            ftype = self.get_fortran_type_for_user_type(sym_kind.identifier)
            InitializationEmitter(self)(ftype, self.name_manager[name], {})

    def emit_variable_deinit(self, name, sym_kind):
        fortran_name = self.name_manager[name]
        refcnt_name = self.name_manager.name_refcount(name)

        from dagrt.data import UserType
        if not isinstance(sym_kind, UserType):
            return

        self.emit(
                "call {var_deinit_name}({args})"
                .format(
                    var_deinit_name=self.get_var_deinit_name(
                        sym_kind.identifier),
                    args=", ".join(
                        self.extra_arguments
                        + (fortran_name, refcnt_name))
                    ))

    def emit_refcounted_allocation(self, sym, sym_kind):
        fortran_name = self.name_manager[sym]

        from dagrt.data import UserType
        if not isinstance(sym_kind, UserType):
            return

        ftype = self.get_fortran_type_for_user_type(sym_kind.identifier)
        AllocationEmitter(self)(ftype, fortran_name, {})

        self.emit_allocate_refcount(self.name_manager.name_refcount(sym))

    def emit_allocate_refcount(self, refcnt_name):
        self.emit_traceable("allocate({name}, stat=dagrt_ierr)".format(
            name=refcnt_name))
        with FortranIfEmitter(self.emitter, "dagrt_ierr.ne.0", self):
            self.emit("write(dagrt_stderr,*) 'failed to allocate {name}'".format(
                name=refcnt_name))
            self.emit("stop")

        self.emit_traceable(f"{refcnt_name} = 1")

    def emit_allocation_check(self, sym, sym_kind):
        fortran_name = self.name_manager[sym]
        refcnt_name = self.name_manager.name_refcount(sym)

        self.emit(
                "call {alloc_check_name}({args})"
                .format(
                    alloc_check_name=self.get_alloc_check_name(
                        sym_kind.identifier),
                    args=", ".join(
                        self.extra_arguments
                        + (fortran_name, refcnt_name))
                    ))

    def emit_user_type_move(self, assignee_sym, assignee_fortran_name,
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
        self.emit("")

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

            from dagrt.data import UserType
            if not isinstance(sym_kind, UserType):
                self.emit(
                        "{name}{subscript_str} = {expr}"
                        .format(
                            name=assignee_fortran_name,
                            subscript_str=subscript_str,
                            expr=self.expr(expr)))
            else:
                ftype = self.get_fortran_type_for_user_type(sym_kind.identifier)
                AssignmentEmitter(self)(
                        ftype, assignee_fortran_name, {}, expr,
                        is_rhs_target=True)

        self.emit("")

    def emit_extra_arg_decl(self, emitter=None):
        if emitter is None:
            emitter = self.emit

        if not self.extra_argument_decl:
            return

        for line in self.extra_argument_decl:
            emitter(line)

        self.emit("")

    # }}}

    # {{{ emit_initialize

    def emit_initialize(self, dag):
        init_symbols = sorted(
                sym
                for sym in self.sym_kind_table.global_table
                if not sym.startswith("<ret"))

        args = self.extra_arguments + ("dagrt_state",) + tuple(
                self.name_manager.name_global(sym)
                for sym in init_symbols)

        function_name = "initialize"
        phase_id = "<dagrt>"+function_name
        self.emit_def_begin(function_name, args, phase_id=phase_id)

        for sym in init_symbols:
            sym_kind = self.sym_kind_table.global_table[sym]
            fortran_name = self.name_manager.name_global(sym)
            self.sym_kind_table.set(phase_id, "<target>"+fortran_name, sym_kind)

        self.declaration_emitter("type(dagrt_state_type), pointer :: dagrt_state")
        self.declaration_emitter("")

        self.current_function = phase_id

        for sym in init_symbols:
            sym_kind = self.sym_kind_table.global_table[sym]
            fortran_name = self.name_manager.name_global(sym)

            self.emit_variable_decl(
                    fortran_name, sym_kind, is_argument=True,
                    other_specifiers=("optional",))

        self.emit("")

        self.emit("character :: dagrt_nan_str*3")
        self.emit("real :: dagrt_nan")

        self.emit("dagrt_nan_str = 'NaN'")
        self.emit("read(dagrt_nan_str,*) dagrt_nan")

        self.emit(
                "dagrt_state%dagrt_next_phase = dagrt_phase_{}"
                .format(dag.initial_phase))

        for sym, sym_kind in sorted(self.sym_kind_table.global_table.items()):
            self.emit_variable_init(sym, sym_kind)

        # {{{ initialize scalar outputs to NaN

        self.emit("")
        self.emit("! initialize scalar outputs to NaN")
        self.emit("")

        for sym in sorted(self.sym_kind_table.global_table):
            sym_kind = self.sym_kind_table.global_table[sym]

            tgt_fortran_name = self.name_manager[sym]

            # All our scalars are floating-point numbers for now,
            # so initializing them all to NaN is fine.

            from dagrt.data import Scalar
            if sym.startswith("<ret") and isinstance(sym_kind, Scalar):
                self.emit("{fortran_name} = dagrt_nan".format(
                    fortran_name=tgt_fortran_name))

        self.emit("")

        # }}}

        for sym in init_symbols:
            sym_kind = self.sym_kind_table.global_table[sym]

            tgt_fortran_name = self.name_manager[sym]
            fortran_name = self.name_manager.name_global(sym)

            with FortranIfEmitter(
                    self.emitter, "present(%s)" % fortran_name, self):
                self.emit_refcounted_allocation(sym, sym_kind)

                from dagrt.data import UserType
                if not isinstance(sym_kind, UserType):
                    self.emit(
                            "{lhs} = {rhs}"
                            .format(
                                lhs=tgt_fortran_name,
                                rhs=fortran_name))
                else:
                    ftype = self.get_fortran_type_for_user_type(
                            sym_kind.identifier)

                    from pymbolic import var
                    AssignmentEmitter(self)(ftype, tgt_fortran_name, {},
                            var("<target>"+fortran_name),
                            is_rhs_target=True)

        # {{{ instrumentation

        if self.emit_instrumentation:
            self.emit("")
            self.emit("! {{{ instrumentation")
            self.emit("")

            for phase_name in sorted(dag.phases):
                self.emit("dagrt_state%%dagrt_phase_%s_count = 0" % phase_name)
                self.emit("dagrt_state%%dagrt_phase_%s_failures = 0" % phase_name)
                self.emit("dagrt_state%%dagrt_phase_%s_time = 0" % phase_name)

            self.emit("")

            for func_name in self.get_called_function_names(dag):
                func_id = make_identifier_from_name(func_name)
                self.emit("dagrt_state%% dagrt_func_%s_count = 0" % func_id)
                self.emit("dagrt_state%%dagrt_func_%s_time = 0" % func_id)

            self.emit("")
            self.emit("! }}}")
            self.emit("")

        # }}}

        self.emit_def_end(function_name)

        self.current_function = None

    # }}}

    # {{{ emit_shutdown

    def emit_shutdown(self):
        args = self.extra_arguments + ("dagrt_state",)

        function_name = "shutdown"
        phase_id = "<dagrt>"+function_name
        self.emit_def_begin(function_name, args, phase_id=phase_id)

        self.declaration_emitter("type(dagrt_state_type), pointer :: dagrt_state")
        self.declaration_emitter("")

        self.current_function = phase_id

        from dagrt.data import UserType

        for sym, sym_kind in sorted(self.sym_kind_table.global_table.items()):
            self.emit_variable_deinit(sym, sym_kind)

        for sym, sym_kind in sorted(self.sym_kind_table.global_table.items()):
            if isinstance(sym_kind, UserType):
                fortran_name = self.name_manager[sym]
                with FortranIfEmitter(
                        self.emitter,
                        f"associated({fortran_name})", self):
                    self.emit(
                            "write(dagrt_stderr,*) 'leaked reference in {name}'"
                            .format(name=fortran_name))
                    self.emit(
                            "write(dagrt_stderr,*) 'remaining refcount ', {name}"
                            .format(name=self.name_manager.name_refcount(sym)))

        self.emit_def_end(function_name)

        self.current_function = None

    # }}}

    # {{{ emit_run_step

    def emit_run_step(self, dag):
        args = self.extra_arguments + ("dagrt_state",)

        function_name = "run"
        phase_id = "<dagrt>"+function_name
        self.emit_def_begin(function_name, args, phase_id=phase_id)

        self.declaration_emitter("type(dagrt_state_type), pointer :: dagrt_state")
        self.declaration_emitter("")

        self.current_function = phase_id

        if_emit = None
        for name, phase_descr in sorted(dag.phases.items()):
            phase_sym_name = self.phase_name_to_phase_sym(name)
            cond = "dagrt_state%dagrt_next_phase == "+phase_sym_name

            if if_emit is None:
                if_emit = FortranIfEmitter(
                        self.emitter, cond, self)
                if_emit.__enter__()
            else:
                if_emit.emit_else_if(cond)

            self.emit(
                    "dagrt_state%dagrt_next_phase = "
                    + self.phase_name_to_phase_sym(phase_descr.next_phase))

            self.emit(
                    "call dagrt_phase_func_{phase_name}({args})".format(
                        phase_name=name,
                        args=", ".join(args)))

        if if_emit:
            if_emit.emit_else()
            self.emit("write(dagrt_stderr,*) 'encountered invalid phase in run', "
                    "dagrt_state%dagrt_next_phase")
            self.emit("stop")

        if_emit.__exit__(None, None, None)

        self.emit_def_end(function_name)

        self.current_function = None

    # }}}

    # {{{ emit_print_profile

    def emit_print_profile(self, dag):
        args = ("dagrt_state",)

        function_name = "print_profile"
        self.emit_def_begin(function_name, args, with_extra_args=False)

        self.declaration_emitter("type(dagrt_state_type), pointer :: dagrt_state")
        self.declaration_emitter("")

        if self.emit_instrumentation:
            delim = "-" * 75
            self.emit("write(*,*) '%s'" % delim)
            self.emit("write(*,*) 'dagrt profile information'")
            self.emit("write(*,*) '%s'" % delim)

            for phase_name in sorted(dag.phases):
                self.emit(
                    "write(*,*) 'phase {phase} count:', "
                    "dagrt_state%dagrt_phase_{phase}_count"
                    .format(phase=phase_name))
                self.emit(
                    "write(*,*) 'phase {phase} failures:', "
                    "dagrt_state%dagrt_phase_{phase}_failures"
                    .format(phase=phase_name))
                with FortranIfEmitter(
                        self.emitter,
                        "dagrt_state%dagrt_phase_{phase}_count > 0"
                        .format(phase=phase_name),
                        self):
                    self.emit(
                        "write(*,*) 'phase {phase} mean time:', "
                        "dagrt_state%dagrt_phase_{phase}_time"
                        "/dagrt_state%dagrt_phase_{phase}_count"
                        .format(phase=phase_name))
                self.emit(
                    "write(*,*) 'phase {phase} total time:', "
                    "dagrt_state%dagrt_phase_{phase}_time"
                    .format(phase=phase_name))

            self.emit("")
            self.emit("write(*,*) '%s'" % delim)
            self.emit("")

            for func_name in self.get_called_function_names(dag):
                func_id = make_identifier_from_name(func_name)
                self.emit(
                    "write(*,*) 'function {func_name} count:', "
                    "dagrt_state%dagrt_func_{func_id}_count"
                    .format(func_name=func_name, func_id=func_id))

                with FortranIfEmitter(
                        self.emitter,
                        "dagrt_state%dagrt_func_{func_id}_count > 0"
                        .format(func_id=func_id),
                        self):
                    self.emit(
                        "write(*,*) 'function {func_name} mean time:', "
                        "dagrt_state%dagrt_func_{func_id}_time"
                        "/dagrt_state%dagrt_func_{func_id}_count"
                        .format(func_name=func_name, func_id=func_id))

                self.emit(
                    "write(*,*) 'function {func_name} total time:', "
                    "dagrt_state%dagrt_func_{func_id}_time"
                    .format(func_name=func_name, func_id=func_id))

            self.emit("write(*,*) '%s'" % delim)

        self.emit_def_end(function_name)

    # }}}

    # {{{ emit_dagrt_function

    def emit_dagrt_function(self, fortran_name, function_id, arg_kinds):
        function = self.function_registry[function_id]

        arg_kinds_dict = dict(zip(function.arg_names, arg_kinds))

        result_kinds = function.get_result_kinds(arg_kinds_dict, check=True)

        result_names = [self.name_manager.make_unique_fortran_name("res%d" % (i + 1))
                for i in range(len(result_kinds))]
        args = (
            list(self.extra_arguments)
            + ["dagrt_state"]
            + list(function.arg_names)
            + result_names)

        self.emit_def_begin(fortran_name, args)

        self.declaration_emitter("type(dagrt_state_type), pointer :: dagrt_state")
        self.declaration_emitter("")

        for name, arg_kind in zip(function.arg_names, arg_kinds):
            if arg_kind is None:
                # We may encounter None as an arg_kind, for arguments of
                # state update notification.
                self.declaration_emitter("integer "+name)
            else:
                self.emit_variable_decl(name, arg_kind, is_argument=True)

        for name, res_kind in zip(result_names, result_kinds):
            self.emit_variable_decl(name, res_kind, is_argument=True)

        self.emit("")

        # {{{ instrumentation

        if self.emit_instrumentation:
            self.emit(
                    "dagrt_state%dagrt_func_{func}_count "
                    "= dagrt_state%dagrt_func_{func}_count + 1"
                    .format(func=make_identifier_from_name(function_id)))

            timer_start_var = self.name_manager.make_unique_fortran_name(
                "timer_start")
            self.declaration_emitter("real*8 " + timer_start_var)

            self.emit(
                    "{timer_start_var} = {timing_function}()"
                    .format(
                        timer_start_var=timer_start_var,
                        timing_function=self.timing_function))

        # }}}

        func_codegen = function.get_codegen(self.language)

        func_codegen(
                results=result_names,
                function=function,
                args=function.arg_names,
                arg_kinds=arg_kinds,
                code_generator=self)

        # {{{ instrumentation

        if self.emit_instrumentation:
            self.emit(
                    "dagrt_state%dagrt_func_{func}_time "
                    "= dagrt_state%dagrt_func_{func}_time "
                    "+ ({timing_function}() - {timer_start_var})"
                    .format(
                        func=make_identifier_from_name(function_id),
                        timing_function=self.timing_function,
                        timer_start_var=timer_start_var,
                        ))

        # }}}

        self.emit_def_end(fortran_name)

        self.current_function = None

    # }}}

    # {{{ called by superclass

    def emit_def_begin(self, function_name, argument_names, phase_id=None,
            with_extra_args=True):
        self.declaration_emitter = FortranEmitter()

        FortranSubroutineEmitter(
                self.emitter,
                function_name,
                argument_names,
                self).__enter__()

        self.declaration_emitter("implicit none")
        self.declaration_emitter("integer dagrt_ierr")
        self.declaration_emitter("")

        if with_extra_args:
            self.emit_extra_arg_decl(self.declaration_emitter)

        body_emitter = FortranEmitter()
        self.emitters.append(body_emitter)

        if phase_id is not None:
            sym_table = self.sym_kind_table.per_phase_table.get(phase_id, {})
            for identifier, sym_kind in sorted(sym_table.items()):
                self.emit_variable_decl(
                        self.name_manager[identifier], sym_kind,
                        refcount_name=self.name_manager.name_refcount(
                                identifier, qualified_with_state=False))

            if sym_table:
                self.emit("")

        self.emit_trace("================================================")
        self.emit_trace("enter %s" % function_name)

        if phase_id is not None:
            for identifier, sym_kind in sorted(sym_table.items()):
                self.emit_variable_init(identifier, sym_kind)

            if sym_table:
                self.emit("")

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
        self.emitter.emit_else()  # pylint:disable=no-member

    def emit_for_begin(self, loop_var_name, lbound, ubound):
        em = FortranDoEmitter(
                self.emitter,
                self.name_manager[loop_var_name],
                "int({}), int({})".format(
                    self.expr(lbound),
                    self.expr(ubound-1)),
                code_generator=self)
        em.__enter__()

    def emit_for_end(self, loop_var_name):
        self.emitter.__exit__(None, None, None)

    def emit_assign_expr(self, assignee_sym, assignee_subscript, expr):
        from dagrt.data import UserType, Array

        assignee_fortran_name = self.name_manager[assignee_sym]

        sym_kind = self.sym_kind_table.get(
                self.current_function, assignee_sym)

        if assignee_subscript and not isinstance(sym_kind, Array):
            raise TypeError("only arrays support subscripted assignment")
            return

        if not isinstance(sym_kind, UserType):
            self.emit_assign_expr_inner(
                    assignee_fortran_name, assignee_subscript, expr, sym_kind)
            return

        if assignee_subscript:
            raise ValueError("User types do not support subscripting")

        if isinstance(expr, Variable):
            self.emit_user_type_move(
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

    def lower_inst(self, inst):
        """Emit the code for an statement."""

        self.emit("! {{{ %s" % inst)
        self.emit("")
        super().lower_inst(inst)
        self.emit("")
        self.emit("! }}}")
        self.emit("")

    # {{{ emit_inst_Assign

    def emit_inst_Assign(self, inst):
        assert not inst.loops

        self.emit_assign_expr(
                inst.assignee, inst.assignee_subscript, inst.expression)
        self.emit_deinit_for_last_usage_of_vars(inst)

    # }}}

    # {{{ emit_inst_AssignFunctionCall

    def emit_inst_AssignFunctionCall(self, inst):
        self.emit_trace("func call {results} = {expr}..."
                .format(
                    results=", ".join(inst.assignees),
                    expr=str(inst.as_expression())[:50]))

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
        from dagrt.data import UserType

        for assignee_sym in inst.assignees:
            sym_kind = self.sym_kind_table.get(
                    self.current_function, assignee_sym)
            if isinstance(sym_kind, UserType):
                self.emit_allocation_check(assignee_sym, sym_kind)

            assert var(assignee_sym) not in DependencyMapper()(
                    inst.as_expression())

        assignee_fortran_names = [
                self.name_manager[assignee_sym] for assignee_sym in inst.assignees]

        function = self.function_registry[inst.function_id]

        arg_kinds = function.resolve_args(arg_kinds_dict)

        key = (inst.function_id, arg_kinds)

        try:
            fortran_func_name = self.function_and_arg_kinds_to_fortran_name[key]
        except KeyError:
            fortran_func_name = self.name_manager.make_unique_fortran_name(
                    inst.function_id)
            self.function_and_arg_kinds_to_fortran_name[key] = fortran_func_name

        self.emit("call {fortran_func_name}({args})"
                .format(
                    fortran_func_name=fortran_func_name,
                    args=", ".join(
                        list(self.extra_arguments)
                        + ["dagrt_state"]
                        + list(function.resolve_args(arg_strs_dict))
                        + assignee_fortran_names
                        )))

        self.emit_deinit_for_last_usage_of_vars(inst)

    # }}}

    def emit_return(self):
        self.emit("goto 999")

    def emit_inst_YieldState(self, inst):
        self.emit_assign_expr(
                "<ret_time_id>"+inst.component_id,
                (),
                Variable("dagrt_time_"+str(inst.time_id)))
        self.emit_assign_expr(
                "<ret_time>"+inst.component_id,
                (),
                inst.time)

        from dagrt.language import AssignFunctionCall
        from pymbolic import var

        if self.call_before_state_update:
            self.emit_inst_AssignFunctionCall(
                    AssignFunctionCall(
                        (),
                        self.call_before_state_update,
                        (var(self.component_name_to_component_sym(
                            inst.component_id)),)))

        self.emit_assign_expr(
                "<ret_state>"+inst.component_id,
                (),
                inst.expression)

        if self.call_after_state_update:
            self.emit_inst_AssignFunctionCall(
                    AssignFunctionCall(
                        (),
                        self.call_after_state_update,
                        (var(self.component_name_to_component_sym(
                            inst.component_id)),)))

    def emit_deinit_for_last_usage_of_vars(self, inst):
        """Check if, for any of the variables in instruction *inst*,
        *inst* contains the last use of that variable in the
        :attr:`current_function`. If so, emit code to deallocate that variable.
        """
        from dagrt.utils import is_state_variable

        read_and_written = inst.get_read_variables() | inst.get_written_variables()

        for variable in read_and_written:
            # FIXME: This can fail for args of state update notification,
            # hence the try/catch.
            try:
                var_kind = self.sym_kind_table.get(
                    self.current_function, variable)
            except KeyError:
                continue

            last_used_stmt_id = self.last_used_stmt_table[
                    variable, self.current_function]
            if inst.id == last_used_stmt_id and not is_state_variable(variable):
                self.emit_variable_deinit(variable, var_kind)

    def emit_inst_Raise(self, inst):
        # FIXME: Reenable emitting full error message
        # TBD: Quoting of quotes, extra-long lines
        if inst.error_message:
            self.emit("! " + inst.error_message)

        self.emit("write (dagrt_stderr,*) "
                "'{condition}'".format(condition=inst.error_condition.__name__))
        self.emit("stop")

    def emit_inst_FailStep(self, inst):
        if self.emit_instrumentation:
            self.emit(
                    "dagrt_state%dagrt_phase_{phase}_failures "
                    "= dagrt_state%dagrt_phase_{phase}_failures + 1"
                    .format(phase=self.current_function))

        self.emit("goto 999")

    def emit_inst_SwitchPhase(self, inst):
        self.emit(
                "dagrt_state%dagrt_next_phase = "
                + self.phase_name_to_phase_sym(inst.next_phase))
        self.emit("goto 999")

    # }}}

# }}}


# {{{ built-in functions

class TypeVisitorWithResult(CodeGeneratingTypeVisitor):
    def __init__(self, code_generator, result_expr):
        super().__init__(code_generator)
        self.result_expr = result_expr


class Norm2Computer(TypeVisitorWithResult):
    def visit_BuiltinType(self, fortran_type, fortran_expr_str, index_expr_map):
        self.code_generator.emit(
                "{result} = {result} + abs({expr})**2"
                .format(
                    result=self.result_expr,
                    expr=fortran_expr_str))


def codegen_builtin_norm_2(results, function, args, arg_kinds,
        code_generator):
    result, = results

    from dagrt.data import Scalar, UserType, Array
    x_kind = arg_kinds[0]
    if isinstance(x_kind, Scalar):
        if x_kind.is_real_valued:
            ftype = BuiltinType("real*8")
        else:
            ftype = BuiltinType("complex*16")
    elif isinstance(x_kind, UserType):
        ftype = code_generator.user_type_map[x_kind.identifier]

    elif isinstance(x_kind, Array):
        code_generator.emit("{result} = norm2({arg})".format(
            result=result, arg=args[0]))
        return

    else:
        raise TypeError("unsupported kind for norm_2 argument: %s" % x_kind)

    code_generator.emit(f"{result} = 0")
    code_generator.emit("")

    Norm2Computer(code_generator, result)(ftype, args[0], {})

    code_generator.emit("")
    code_generator.emit("{result} = sqrt({result})".format(result=result))
    code_generator.emit("")


class LenComputer(TypeVisitorWithResult):
    # FIXME: This could be made *way* more efficient by handling
    # arrays of built-in types directly.

    def visit_BuiltinType(self, fortran_type, fortran_expr_str, index_expr_map):
        self.code_generator.emit(
                "{result} = {result} + 1"
                .format(
                    result=self.result_expr))


def codegen_builtin_len(results, function, args, arg_kinds,
        code_generator):
    result, = results

    from dagrt.data import Scalar, Array, UserType
    x_kind = arg_kinds[0]
    if isinstance(x_kind, Scalar):
        if x_kind.is_real_valued:
            ftype = BuiltinType("real*8")
        else:
            ftype = BuiltinType("complex*16")
    elif isinstance(x_kind, UserType):
        ftype = code_generator.user_type_map[x_kind.identifier]
    elif isinstance(x_kind, Array):
        code_generator.emit("{result} = size({arg})".format(
            result=result,
            arg=args[0]))
        return
    else:
        raise TypeError("unsupported kind for norm_2 argument: %s" % x_kind)

    code_generator.emit(f"{result} = 0")
    code_generator.emit("")

    LenComputer(code_generator, result)(ftype, args[0], {})
    code_generator.emit("")


class IsNaNComputer(TypeVisitorWithResult):
    def visit_BuiltinType(self, fortran_type, fortran_expr_str, index_expr_map):
        self.code_generator.emit(
                "{result} = {result} .or. (({expr}).ne.({expr}))"
                .format(
                    result=self.result_expr,
                    expr=fortran_expr_str))


def codegen_builtin_isnan(results, function, args, arg_kinds,
        code_generator):
    result, = results

    from dagrt.data import Scalar, UserType
    x_kind = arg_kinds[0]
    if isinstance(x_kind, Scalar):
        if x_kind.is_real_valued:
            ftype = BuiltinType("real*8")
        else:
            ftype = BuiltinType("complex*16")
    elif isinstance(x_kind, UserType):
        ftype = code_generator.user_type_map[x_kind.identifier]
    else:
        raise TypeError("unsupported kind for norm_2 argument: %s" % x_kind)

    code_generator.emit(f"{result} = .false.")
    code_generator.emit("")

    IsNaNComputer(code_generator, result)(ftype, args[0], {})
    code_generator.emit("")


builtin_array = CallCode("""
        if (int(${n}).ne.${n}) then
            write(dagrt_stderr,*) 'argument to array() is not an integer'
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
            write(dagrt_stderr,*) &
                'argument ' // &
                '${cols_var} ' // &
                'to ${func_name}' // &
                'is not an integer'
            stop
        endif

        ${rows_var} = size(${mat_array}) / int(${cols_var})

        if (${rows_var} * int(${cols_var}) .ne. size(${mat_array})) then
            write(dagrt_stderr,*) &
                'size of argument ' // &
                '${mat_array} ' // &
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


builtin_transpose = CallCode(UTIL_MACROS + """
        <%
        a_rows = declare_new("integer", "a_rows")
        res_size = declare_new("integer", "res_size")
        %>

        ${check_matrix(a, a_cols, a_rows, "transpose")}

        ${a_rows} = size(${a}) / int(${a_cols})
        ${res_size} = ${a_rows} * int(${a_cols})

        if (allocated(${result})) then
            deallocate(${result})
        endif

        allocate(${result}(0:${res_size}-1))

        ${result} = reshape( &
                transpose( &
                    reshape(${a}, (/${a_rows}, int(${a_cols})/))), &
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
            write(dagrt_stderr,*) 'inconsistent matrix sizes in linear_solve'
            stop
        endif
        if (int(${a_rows}).ne.int(${a_cols})) then
            write(dagrt_stderr,*) 'non-square matrix sizes in linear_solve'
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
            write(dagrt_stderr,*) &
                'gesv on ${a} failed with info=', ${info}
            stop
        endif

        deallocate(${lu_temp})
        deallocate(${ipiv})

        """)


builtin_svd = CallCode(UTIL_MACROS + """
        <%
        sigma_size = declare_new("integer", "res_size")
        a_rows = declare_new("integer", "a_rows")

        %>

        ${check_matrix(a, a_cols, a_rows, "svd")}
        ${sigma_size} = min(int(${a_cols}),int(${a_rows}))

        <%
        ltr = get_lapack_letter(a_kind)

        a_temp = declare_new(
                kind_to_fortran(a_kind)+", dimension(:), allocatable"
                , "a_temp")
        work = declare_new(
                kind_to_fortran(a_kind)+", dimension(:), allocatable"
                , "work")
        info = declare_new("integer", "info")
        lwork = declare_new("integer", "lwork")
        lda = declare_new("integer", "lda")
        ldu = declare_new("integer", "ldu")
        ldvt = declare_new("integer", "ldvt")
        jobu = declare_new("character*1", "jobu")
        jobvt = declare_new("character*1", "jobvt")
        %>

        allocate(${a_temp}(0:size(${a})-1))
        ${jobu} = "S"
        ${jobvt} = "S"
        ${lda} = max(1,int(${a_rows}))
        ${ldu} = int(${a_rows})
        ${ldvt} = min(int(${a_rows}),int(${a_rows}))

        ${a_temp} = ${a}
        ${lwork} = max(1, &
                3*min(int(${a_rows}), &
                int(${a_cols})) + max(int(${a_rows}), &
                int(${a_cols})), &
                5*min(int(${a_rows}), int(${a_cols})))

        if (allocated(${sigma})) then
            deallocate(${sigma})
        endif

        allocate(${sigma}(0:${sigma_size}-1))
        allocate(${work}(0:${lwork}-1))
        allocate(${u}(0:int(${a_rows}*${a_rows})-1))
        allocate(${vt}(0:int(${a_rows}*${a_cols})-1))

        call ${ltr}gesvd(${jobu}, ${jobvt}, &
            int(${a_rows}), int(${a_cols}), ${a_temp}, ${lda}, ${sigma}, &
            ${u}, ${ldu}, ${vt}, ${ldvt}, ${work}, ${lwork}, ${info})

        if (${info}.ne.0) then
            write(dagrt_stderr,*) &
                'gesvd on ${a} failed with info=', ${info}
            stop
        endif

        deallocate(${a_temp})
        deallocate(${work})

        """)


builtin_print = CallCode(UTIL_MACROS + """
        write(*,*) ${arg}
        """)

# }}}


# vim: foldmethod=marker
