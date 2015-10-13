from __future__ import division, with_statement

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2014 Matt Wala
"""

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

from pytools import RecordWithoutPickling
from dagrt.vm.codegen.data import (
        ODEComponent, Integer, Boolean, Scalar, Array, UnableToInferKind)

NoneType = type(None)


# {{{ function

class FunctionNotFound(KeyError):
    pass


class Function(RecordWithoutPickling):
    """
    .. attribute:: result_names

        A list of names of the return values of the function. Note that these
        names serve as documentation and as identifiers to be used for
        variables receiving function results inside generated code implementing
        the call to the function (e.g. in the Fortran backend). They have
        no relationship to the names to which the results ultimately get
        assigned.

    .. attribute:: identifier
    .. attribute:: arg_names
    .. attribute:: default_dict
    """

    def __init__(self, language_to_codegen=None, **kwargs):
        if language_to_codegen is None:
            language_to_codegen = {}

        super(Function, self).__init__(
                language_to_codegen=language_to_codegen,
                **kwargs)

    def get_result_kinds(self, arg_kinds, check):
        """Return a tuple of the :class:`leap.vm.codegen.data.SymbolKind`
        instances for the values this function returns if arguments of the
        kinds *arg_kinds* are supplied.

        The length of the returned tuple must match the lenght of
        :attr:`result_names`.

        :arg arg_kinds: a dictionary mapping numbers (for positional arguments)
            or identifiers (for keyword arguments) to
            :class:`dagrt.vm.codegen.data.SymbolKind` instances indicating the
            types of the arguments being passed to the function.
            Some elements of *arg_kinds* may be None if their kinds
            have yet not been determined.

        :arg check: A :class:`bool`. If True, none of *arg_kinds* will
            be None, and argument kind checking should be performed.
        """
        raise NotImplementedError()

    def register_codegen(self, language, codegen_function):
        new_language_to_codegen = self.language_to_codegen.copy()

        if language in new_language_to_codegen:
            raise ValueError("a code generator for function '%s' in "
                    "language '%s' is already known"
                    % (self.identifier, language))

        new_language_to_codegen[language] = codegen_function
        return self.copy(language_to_codegen=new_language_to_codegen)

    def get_codegen(self, language):
        try:
            return self.language_to_codegen[language]
        except KeyError:
            raise KeyError(
                    "'%s' has no code generator for language '%s'"
                    % (self.identifier, language))

    def resolve_args(self, arg_dict):
        from dagrt.vm.utils import resolve_args
        return resolve_args(self.arg_names, self.default_dict, arg_dict)


class FixedResultKindsFunction(Function):
    def __init__(self, **kwargs):
        result_kinds = kwargs.get("result_kinds", None)
        if result_kinds is None:
            raise TypeError("result_kinds argument must be specified")

        super(FixedResultKindsFunction, self).__init__(**kwargs)

    def get_result_kinds(self, arg_kinds, check):
        return self.result_kinds

# }}}


# {{{ function registry

class FunctionRegistry(RecordWithoutPickling):
    def __init__(self, id_to_function=None):
        if id_to_function is None:
            id_to_function = {}

        super(FunctionRegistry, self).__init__(
                id_to_function=id_to_function)

    def register(self, function):
        """Return a copy of *self* with *function* registered."""

        if function.identifier in self.id_to_function:
            raise ValueError("function '%s' is already registered"
                    % function.identifier)

        new_id_to_function = self.id_to_function.copy()
        new_id_to_function[function.identifier] = function
        return self.copy(id_to_function=new_id_to_function)

    def __getitem__(self, function_id):
        try:
            return self.id_to_function[function_id]
        except KeyError:
            raise FunctionNotFound(
                    "unknown function: '%s'"
                    % function_id)

    def __contains__(self, function_id):
        return function_id in self.id_to_function

    def register_codegen(self, function_id, language, codegen):
        """Register a code generation helper object for target *language*
        for the function with identifier *function_id*.

        :arg codegen: an object obeying an interface suitable for code
            generation for *language*. This interface depends on the code
            generator being used.
        """
        func = (self.id_to_function[function_id]
                .register_codegen(language, codegen))

        new_id_to_function = self.id_to_function.copy()
        new_id_to_function[function_id] = func
        return self.copy(id_to_function=new_id_to_function)

    def get_codegen(self, function_id, language):
        try:
            func = self.id_to_function[function_id]
        except KeyError:
            raise FunctionNotFound(
                    "unknown function: '%s'"
                    % function_id)

        return func.get_codegen(language)

# }}}


# {{{ built-in functions

class _NormBase(Function):
    """``norm(x)`` returns the *ord*-norm of *x*."""

    result_names = ("result",)
    identifier = "<builtin>norm"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'norm' is not an ODE component")

        return (Scalar(is_real_valued=True),)


class _Norm1(_NormBase):
    """``norm_1(x)`` returns the 1-norm of *x*."""
    identifier = "<builtin>norm_1"


class _Norm2(_NormBase):
    """``norm_2(x)`` returns the 2-norm of *x*."""
    identifier = "<builtin>norm_2"


class _NormInf(_NormBase):
    """``norm_inf(x)`` returns the infinity-norm of *x*."""
    identifier = "<builtin>norm_inf"


class _DotProduct(Function):
    """dot_product(x, y)`` return the dot product of *x* and *y*. The
    complex conjugate of *x* is taken first, if applicable.
    """

    result_names = ("result",)
    identifier = "<builtin>dot_product"
    arg_names = ("x", "y")
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, y_kind = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'dot_product' is not an ODE component")
        if check and not isinstance(y_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'y' of 'dot_product' is not an ODE component")

        return (Scalar(is_real_valued=False),)


class _Len(Function):
    """``len(x)`` returns the number of degrees of freedom in *x* """

    result_names = ("result",)
    identifier = "<builtin>len"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return (Scalar(is_real_valued=True),)


class _IsNaN(Function):
    """``isnan(x)`` returns True if there are any NaNs in *x*"""

    result_names = ("result",)
    identifier = "<builtin>isnan"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return (Boolean(),)


class _Array(Function):
    """``array(n)`` returns an empty array with n entries in it.
    n must be an integer.
    """

    result_names = ("result",)
    identifier = "<builtin>array"
    arg_names = ("n",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        n_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(n_kind, Scalar):
            raise TypeError("argument 'n' of 'array' is not a scalar")

        return (Array(is_real_valued=True),)


class _MatMul(Function):
    """``matmul(a, b, a_cols, b_cols)`` returns a 1D array containing the
    matrix resulting from multiplying the arrays a and b (both interpreted
    as matrices, with a number of columns *a_cols* and *b_cols* respectively)
    """

    result_names = ("result",)
    identifier = "<builtin>matmul"
    arg_names = ("a", "b", "a_cols", "b_cols")
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        a_kind, b_kind, a_cols_kind, b_cols_kind = self.resolve_args(arg_kinds)

        if a_kind is None or b_kind is None:
            raise UnableToInferKind(
                    "matmul needs to know both arguments to infer result kind")

        if check and not isinstance(a_kind, Array):
            raise TypeError("argument 'a' of 'matmul' is not an array")
        if check and not isinstance(b_kind, Array):
            raise TypeError("argument 'a' of 'matmul' is not an array")
        if check and not isinstance(a_cols_kind, Scalar):
            raise TypeError("argument 'a_cols' of 'matmul' is not a scalar")
        if check and not isinstance(b_cols_kind, Scalar):
            raise TypeError("argument 'b_cols' of 'matmul' is not a scalar")

        is_real_valued = a_kind.is_real_valued and b_kind.is_real_valued

        return (Array(is_real_valued),)


class _LinearSolve(Function):
    """``linear_solve(a, b, a_cols, b_cols)`` returns a 1D array containing the
    matrix resulting from multiplying the matrix inverse of a by b (both interpreted
    as matrices, with a number of columns *a_cols* and *b_cols* respectively)
    """

    result_names = ("result",)
    identifier = "<builtin>linear_solve"
    arg_names = ("a", "b", "a_cols", "b_cols")
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        a_kind, b_kind, a_cols_kind, b_cols_kind = self.resolve_args(arg_kinds)

        if a_kind is None or b_kind is None:
            raise UnableToInferKind(
                    "linear_solve needs to know both arguments to infer result kind")

        if check and not isinstance(a_kind, Array):
            raise TypeError("argument 'a' of 'linear_solve' is not an array")
        if check and not isinstance(b_kind, Array):
            raise TypeError("argument 'a' of 'linear_solve' is not an array")
        if check and not isinstance(a_cols_kind, Scalar):
            raise TypeError("argument 'a_cols' of 'linear_solve' is not a scalar")
        if check and not isinstance(b_cols_kind, Scalar):
            raise TypeError("argument 'b_cols' of 'linear_solve' is not a scalar")

        is_real_valued = a_kind.is_real_valued and b_kind.is_real_valued

        return (Array(is_real_valued),)


class _Print(Function):
    """``print(arg)`` prints the given operand to standard output. Returns an integer
    that may be ignored.
    """

    result_names = ()
    identifier = "<builtin>print"
    arg_names = ("arg",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        arg_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(arg_kind, (Integer, Scalar, Array)):
            raise TypeError(
                    "argument of 'print' is not an integer, array, or scalar")

        return (Integer(),)


class _PythonBuiltinFunctionCodeGenerator(object):
    def __init__(self, function, pattern):
        self.function = function
        self.pattern = pattern

    def __call__(self, expr_mapper, arg_strs_dict):
        args = self.function.resolve_args(arg_strs_dict)
        return self.pattern.format(
            numpy=expr_mapper._numpy,
            args=", ".join(args))


def _make_bfr():
    bfr = FunctionRegistry()

    for func, py_pattern in [
            (_Norm1(), "self._builtin_norm_1({args})"),
            (_Norm2(), "self._builtin_norm_2({args})"),
            (_NormInf(), "self._builtin_norm_inf({args})"),
            (_DotProduct(), "{numpy}.vdot({args})"),
            (_Len(), "{numpy}.size({args})"),
            (_IsNaN(), "{numpy}.isnan({args})"),
            (_Array(), "self._builtin_array({args})"),
            (_MatMul(), "self._builtin_matmul({args})"),
            (_LinearSolve(), "self._builtin_linear_solve({args})"),
            (_Print(), "self._builtin_print({args})"),
            ]:

        bfr = bfr.register(func)
        bfr = bfr.register_codegen(
            func.identifier,
            "python",
            _PythonBuiltinFunctionCodeGenerator(
                func, py_pattern))

    import dagrt.vm.codegen.fortran as f

    bfr = bfr.register_codegen(_Norm2.identifier, "fortran",
            f.codegen_builtin_norm_2)
    bfr = bfr.register_codegen(_Len.identifier, "fortran",
            f.codegen_builtin_len)
    bfr = bfr.register_codegen(_IsNaN.identifier, "fortran",
            f.codegen_builtin_isnan)
    bfr = bfr.register_codegen(_Array.identifier, "fortran",
            f.builtin_array)
    bfr = bfr.register_codegen(_MatMul.identifier, "fortran",
            f.builtin_matmul)
    bfr = bfr.register_codegen(_LinearSolve.identifier, "fortran",
            f.builtin_linear_solve)
    bfr = bfr.register_codegen(_Print.identifier, "fortran",
            f.builtin_print)

    return bfr

base_function_registry = _make_bfr()

# }}}


# {{{ ODE RHS registration

class _ODERightHandSide(Function):
    default_dict = {}

    result_names = ("result",)

    # Explicitly specify the fields of this record, otherwise, the list of fields may
    # be inherited from the superclass if an instance of the superclass is
    # initialized first. We wish to exclude "arg_names" as a field, since this class
    # synthesizes it as a member.
    fields = set(["identifier", "component_id", "input_component_ids",
                  "language_to_codegen", "input_component_names"])

    def __init__(self, identifier, component_id, input_component_ids,
            language_to_codegen=None, input_component_names=None):
        if input_component_names is None:
            input_component_names = input_component_ids

        super(_ODERightHandSide, self).__init__(
                identifier=identifier,
                component_id=component_id,
                input_component_ids=input_component_ids,
                language_to_codegen=language_to_codegen,
                input_component_names=input_component_names)

    @property
    def arg_names(self):
        return ("t",) + self.input_component_names

    def get_result_kinds(self, arg_kinds, check):
        arg_kinds = self.resolve_args(arg_kinds)

        if check and not isinstance(arg_kinds[0], Scalar):
            raise TypeError("argument 't' of '%s' is not a scalar"
                    % self.identifier)

        for arg_name, arg_kind_passed, input_component_id in zip(
                self.arg_names[1:], arg_kinds[1:], self.input_component_ids):
            if arg_kind_passed is None and not check:
                pass
            elif check and not (isinstance(arg_kind_passed, ODEComponent)
                    and arg_kind_passed.component_id == input_component_id):
                raise TypeError("argument '%s' of '%s' is not an ODE component "
                        "with component ID '%s'"
                        % (arg_name, self.identifier, input_component_id))

        return (ODEComponent(self.component_id),)


def register_ode_rhs(
        function_registry,
        component_id, identifier=None, input_component_ids=None,
        input_component_names=None):
    if identifier is None:
        identifier = "<func>"+component_id

    if input_component_ids is None:
        input_component_ids = (component_id,)

    return function_registry.register(
            _ODERightHandSide(
                identifier, component_id, input_component_ids,
                input_component_names=input_component_names))


def register_function(
        function_registry,
        identifier,
        arg_names,
        default_dict=None,
        result_names=(),
        result_kinds=()):

    return function_registry.register(
            FixedResultKindsFunction(
                identifier=identifier,
                arg_names=arg_names,
                default_dict=default_dict,
                result_names=result_names,
                result_kinds=result_kinds))

# }}}

# vim: foldmethod=marker
