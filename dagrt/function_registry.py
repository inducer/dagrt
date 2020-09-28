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
from dagrt.data import (
        UserType, Integer, Boolean, Scalar, Array, UnableToInferKind)

NoneType = type(None)

__doc__ = """
The function registry is used by targets to resolve external functions and
invoke user-specified code, including but not limited to ODE right-hand sides.

.. autoclass:: Function
.. autoclass:: FunctionRegistry
.. autoclass:: FunctionNotFound

.. data:: base_function_registry

The default function registry, containing all the built-in functions (see
:ref:`built-ins`).

Registering new functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: register_ode_rhs
.. autofunction:: register_function

.. _built-ins:

Built-ins
^^^^^^^^^

The built-in functions are listed below. This also serves as their language
documentation.

.. autoclass:: Norm1
.. autoclass:: Norm2
.. autoclass:: NormInf
.. autoclass:: DotProduct
.. autoclass:: Len
.. autoclass:: IsNaN
.. autoclass:: Array_
.. autoclass:: MatMul
.. autoclass:: Transpose
.. autoclass:: LinearSolve
.. autoclass:: SVD
.. autoclass:: Print

"""


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

        The name of the function.

    .. attribute:: arg_names

        The names of the arguments to the function.

    .. attribute:: default_dict

        A dictionary mapping argument names to default values.

    .. automethod:: get_result_kinds
    .. automethod:: register_codegen
    .. automethod:: get_codegen
    .. automethod:: resolve_args
    """

    def __init__(self, language_to_codegen=None, **kwargs):
        if language_to_codegen is None:
            language_to_codegen = {}

        super().__init__(
                language_to_codegen=language_to_codegen,
                **kwargs)

    def get_result_kinds(self, arg_kinds, check):
        """Return a tuple of the :class:`dagrt.data.SymbolKind`
        instances for the values this function returns if arguments of the
        kinds *arg_kinds* are supplied.

        The length of the returned tuple must match the lenght of
        :attr:`result_names`.

        :arg arg_kinds: a dictionary mapping numbers (for positional arguments)
            or identifiers (for keyword arguments) to
            :class:`dagrt.data.SymbolKind` instances indicating the
            types of the arguments being passed to the function.
            Some elements of *arg_kinds* may be None if their kinds
            have yet not been determined.

        :arg check: A :class:`bool`. If True, none of *arg_kinds* will
            be None, and argument kind checking should be performed.
        """
        raise NotImplementedError()

    def register_codegen(self, language, codegen_function):
        """Return a copy of *self* with *codegen_function*
        registered as a code generator for *language*.

        The interface for *codegen_function* depends on the
        code generator being used.
        """
        new_language_to_codegen = self.language_to_codegen.copy()

        if language in new_language_to_codegen:
            raise ValueError("a code generator for function '%s' in "
                    "language '%s' is already known"
                    % (self.identifier, language))

        new_language_to_codegen[language] = codegen_function
        return self.copy(language_to_codegen=new_language_to_codegen)

    def get_codegen(self, language):
        """Return the code generator for *language*.
        """
        try:
            return self.language_to_codegen[language]
        except KeyError:
            raise KeyError(
                    "'%s' has no code generator for language '%s'"
                    % (self.identifier, language))

    def resolve_args(self, arg_dict):
        """Resolve positional and keyword arguments to an argument list.

        See also :func:`dagrt.utils.resolve_args`.

        :arg arg_dict: a dictionary mapping numbers (for positional arguments)
            or identifiers (for keyword arguments) to values
        """
        from dagrt.utils import resolve_args
        return resolve_args(self.arg_names, self.default_dict, arg_dict)


class FixedResultKindsFunction(Function):
    def __init__(self, **kwargs):
        result_kinds = kwargs.get("result_kinds", None)
        if result_kinds is None:
            raise TypeError("result_kinds argument must be specified")

        super().__init__(**kwargs)

    def get_result_kinds(self, arg_kinds, check):
        return self.result_kinds

# }}}


# {{{ function registry

class FunctionRegistry(RecordWithoutPickling):
    """
    .. automethod:: register
    .. automethod:: __getitem__
    .. automethod:: __contains__
    .. automethod:: register_codegen
    .. automethod:: get_codegen
    """
    def __init__(self, id_to_function=None):
        if id_to_function is None:
            id_to_function = {}

        super().__init__(
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
        """Return the :class:`Function` with identifier *function_id*.

        :raises FunctionNotFound: when *function_id* was not found
        """
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

        if check and not isinstance(x_kind, (NoneType, Array, UserType)):
            raise TypeError("argument 'x' of 'norm' is not a user type")

        return (Scalar(is_real_valued=True),)


class Norm1(_NormBase):
    """``norm_1(x)`` returns the 1-norm of *x*.
    *x* is a user type or array.
    """
    identifier = "<builtin>norm_1"


class Norm2(_NormBase):
    """``norm_2(x)`` returns the 2-norm of *x*.
    *x* is a user type or array.
    """
    identifier = "<builtin>norm_2"


class NormInf(_NormBase):
    """``norm_inf(x)`` returns the infinity-norm of *x*.
    *x* is a user type or array.
    """
    identifier = "<builtin>norm_inf"


class DotProduct(Function):
    """``dot_product(x, y)`` return the dot product of *x* and *y*. The
    complex conjugate of *x* is taken first, if applicable.
    *x* and *y* are either arrays (that must be of the same length) or the
    same user type.
    """

    result_names = ("result",)
    identifier = "<builtin>dot_product"
    arg_names = ("x", "y")
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, y_kind = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, Array, UserType)):
            raise TypeError("argument 'x' of 'dot_product' is not a user type")
        if check and not isinstance(y_kind, (NoneType, Array, UserType)):
            raise TypeError("argument 'y' of 'dot_product' is not a user type")

        return (Scalar(is_real_valued=False),)


class Len(Function):
    """``len(x)`` returns the number of degrees of freedom in *x*.
    *x* is a user type or array.
    """

    result_names = ("result",)
    identifier = "<builtin>len"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, Scalar, Array, UserType)):
            raise TypeError("argument 'x' of 'len' is not a user type")

        return (Scalar(is_real_valued=True),)


class IsNaN(Function):
    """``isnan(x)`` returns True if and only if there are any NaNs in *x*.
    *x* is a user type, scalar, or array.
    """

    result_names = ("result",)
    identifier = "<builtin>isnan"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        x_kind, = self.resolve_args(arg_kinds)

        if check and not isinstance(x_kind, (NoneType, Scalar, Array, UserType)):
            raise TypeError("argument 'x' of 'isnan' is not a user type")

        return (Boolean(),)


class Array_(Function):  # noqa
    """``array(n)`` returns an empty array with *n* entries in it.
    *n* must be an integer.
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


class MatMul(Function):
    """``matmul(a, b, a_cols, b_cols)`` returns a 1D array containing the
    matrix resulting from multiplying the arrays *a* and *b* (both interpreted
    as matrices, with a number of columns *a_cols* and *b_cols* respectively).
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


class Transpose(Function):
    """``transpose(a, a_cols)`` returns a 1D array containing the
    matrix resulting from transposing the array *a* (interpreted
    as a matrix with *a_cols* columns).
    """

    result_names = ("result",)
    identifier = "<builtin>transpose"
    arg_names = ("a", "a_cols")
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        a_kind, a_cols_kind = self.resolve_args(arg_kinds)

        if a_kind is None:
            raise UnableToInferKind(
                    "transpose needs to know both arguments to infer result kind")

        if check and not isinstance(a_kind, Array):
            raise TypeError("argument 'a' of 'transpose' is not an array")
        if check and not isinstance(a_cols_kind, Scalar):
            raise TypeError("argument 'a_cols' of 'transpose' is not a scalar")

        is_real_valued = a_kind.is_real_valued

        return (Array(is_real_valued),)


class LinearSolve(Function):
    """``linear_solve(a, b, a_cols, b_cols)`` returns a 1D array containing the
    matrix resulting from multiplying the matrix inverse of *a* by *b*, both
    interpreted as matrices, with a number of columns *a_cols* and *b_cols*
    respectively.
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


class SVD(Function):
    """``SVD(a, a_cols)`` returns a 2D array ``u``, a 1D array ``sigma``, and
    a 2D array ``vt``, representing the (reduced) SVD of ``a``.
    """

    result_names = ("u", "sigma", "vt")
    identifier = "<builtin>svd"
    arg_names = ("a", "a_cols")
    default_dict = {}

    def get_result_kinds(self, arg_kinds, check):
        a_kind, a_cols_kind = self.resolve_args(arg_kinds)

        if a_kind is None:
            raise UnableToInferKind(
                    "svd needs to know its argument to infer result kind")

        if check and not isinstance(a_kind, Array):
            raise TypeError("argument 'a' of 'svd' is not an array")
        if check and not isinstance(a_cols_kind, Scalar):
            raise TypeError("argument 'a_cols' of 'svd' is not a scalar")

        is_real_valued = a_kind.is_real_valued

        return (Array(is_real_valued), Array(is_real_valued), Array(is_real_valued))


class Print(Function):
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

        return ()


class _PythonBuiltinFunctionCodeGenerator:
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
            (Norm1(), "self._builtin_norm_1({args})"),
            (Norm2(), "self._builtin_norm_2({args})"),
            (NormInf(), "self._builtin_norm_inf({args})"),
            (DotProduct(), "{numpy}.vdot({args})"),
            (Len(), "{numpy}.size({args})"),
            (IsNaN(), "{numpy}.isnan({args})"),
            (Array_(), "self._builtin_array({args})"),
            (MatMul(), "self._builtin_matmul({args})"),
            (Transpose(), "self._builtin_transpose({args})"),
            (LinearSolve(), "self._builtin_linear_solve({args})"),
            (Print(), "self._builtin_print({args})"),
            (SVD(), "self._builtin_svd({args})"),
            ]:

        bfr = bfr.register(func)
        bfr = bfr.register_codegen(
            func.identifier,
            "python",
            _PythonBuiltinFunctionCodeGenerator(
                func, py_pattern))

    import dagrt.codegen.fortran as f

    bfr = bfr.register_codegen(Norm2.identifier, "fortran",
            f.codegen_builtin_norm_2)
    bfr = bfr.register_codegen(Len.identifier, "fortran",
            f.codegen_builtin_len)
    bfr = bfr.register_codegen(IsNaN.identifier, "fortran",
            f.codegen_builtin_isnan)
    bfr = bfr.register_codegen(Array_.identifier, "fortran",
            f.builtin_array)
    bfr = bfr.register_codegen(MatMul.identifier, "fortran",
            f.builtin_matmul)
    bfr = bfr.register_codegen(Transpose.identifier, "fortran",
            f.builtin_transpose)
    bfr = bfr.register_codegen(LinearSolve.identifier, "fortran",
            f.builtin_linear_solve)
    bfr = bfr.register_codegen(SVD.identifier, "fortran",
            f.builtin_svd)
    bfr = bfr.register_codegen(Print.identifier, "fortran",
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
    fields = {"identifier", "output_type_id", "input_type_ids",
                  "language_to_codegen", "input_names"}

    def __init__(self, identifier, output_type_id, input_type_ids,
            language_to_codegen=None, input_names=None):
        if input_names is None:
            input_names = input_type_ids

        super().__init__(
                identifier=identifier,
                output_type_id=output_type_id,
                input_type_ids=input_type_ids,
                language_to_codegen=language_to_codegen,
                input_names=input_names)

    @property
    def arg_names(self):
        return ("t",) + self.input_names

    def get_result_kinds(self, arg_kinds, check):
        arg_kinds = self.resolve_args(arg_kinds)

        if check and not isinstance(arg_kinds[0], Scalar):
            raise TypeError("argument 't' of '%s' is not a scalar"
                    % self.identifier)

        for arg_name, arg_kind_passed, input_usertype_id in zip(
                self.arg_names[1:], arg_kinds[1:], self.input_type_ids):
            if arg_kind_passed is None and not check:
                pass
            elif check and not (isinstance(arg_kind_passed, UserType)
                    and arg_kind_passed.identifier == input_usertype_id):
                raise TypeError("argument '%s' of '%s' is not a user type "
                        "with identifier '%s'"
                        % (arg_name, self.identifier, input_usertype_id))

        return (UserType(self.output_type_id),)


def register_ode_rhs(
        function_registry,
        output_type_id, identifier=None, input_type_ids=None,
        input_names=None):
    """Register a function as an ODE right-hand side.

    Functions registered through this call have the following characteristics.
    First, there is a single return value of the user type whose type identifier
    is *output_type_id*. Second, the function has as its first argument a scalar
    named *t*. Last, the remaining argument list to the function consists of
    user type values.

    For example, considering the ODE :math:`y' = f(t, y)`, the following call
    registers a right-hand side function with name *f* and user type *y*::

        freg = register_ode_rhs(freg, "y", identifier="f")

    :arg function_registry: the base function registry
    :arg output_type_id: a string, the user type ID returned by the call.
    :arg identifier: the full name of the function. If not provided, defaults
       to *<func> + output_type_id*.
    :arg input_type_ids: a tuple of strings, the identifiers of the user types
       which are the arguments to the right-hand side function. An automatically
       added *t* argument occurs before these arguments. If not provided,
       defaults to *(output_type_id,)*.
    :arg input_names: a tuple of strings, the names of the inputs. If not provided,
       defaults to *input_type_ids*.

    :returns: a new :class:`FunctionRegistry`

    """
    if identifier is None:
        identifier = "<func>"+output_type_id

    if input_type_ids is None:
        input_type_ids = (output_type_id,)

    return function_registry.register(
            _ODERightHandSide(
                identifier, output_type_id, input_type_ids,
                input_names=input_names))


def register_function(
        function_registry,
        identifier,
        arg_names,
        default_dict=None,
        result_names=(),
        result_kinds=()):
    r"""Register a function returning output(s) of fixed kind.

    :arg function_registry: the base :class:`FunctionRegistry`
    :arg identifier: a string, the function identifier
    :arg arg_names: a list of strings, the names of the arguments
    :arg default_dict: a dictionary mapping argument names to default
        values
    :arg result_names: a list of strings, the names of the output(s)
    :arg result_kinds: a list of :class:`dagrt.data.SymbolKind`\ s,
        the kinds of the output(s)

    :returns: a new :class:`FunctionRegistry`
    """

    return function_registry.register(
            FixedResultKindsFunction(
                identifier=identifier,
                arg_names=arg_names,
                default_dict=default_dict,
                result_names=result_names,
                result_kinds=result_kinds))

# }}}

# vim: foldmethod=marker
