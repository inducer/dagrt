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
from leap.vm.codegen.data import ODEComponent, Boolean, Scalar

NoneType = type(None)


# {{{ function

class FunctionNotFound(KeyError):
    pass


class Function(RecordWithoutPickling):
    """
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

    def get_result_kind(self, arg_kinds):
        """Return the :class:`leap.vm.codegen.data.SymbolKind` this function
        returns if arguments of the kinds *arg_kinds* are supplied.

        :arg arg_kinds: a dictionary mapping numbers (for positional arguments)
            or identifiers (for keyword arguments) to
            :class:`leap.vm.codegen.data.SymbolKind` instances indicating the
            types of the arguments being passed to the function.
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
        from leap.vm.utils import resolve_args
        return resolve_args(self.arg_names, self.default_dict, arg_dict)

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

class _Norm(Function):
    """norm(x, ord)`` returns the *ord*-norm of *x*."""

    identifier = "<builtin>norm"
    arg_names = ("x", "ord")
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, ord_kind = self.resolve_args(arg_kinds)

        if not isinstance(ord_kind, (NoneType, Scalar)):
            raise TypeError("argument 'ord' of 'norm' is not a scalar")
        if not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'norm' is not an ODE component")

        return Scalar(is_real_valued=True)


class _DotProduct(Function):
    """dot_product(x, y)`` return the dot product of *x* and *y*. The
    complex conjugate of *x* is taken first, if applicable.
    """

    identifier = "<builtin>dot_product"
    arg_names = ("x", "y")
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, y_kind = self.resolve_args(arg_kinds)

        if not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'dot_product' is not an ODE component")
        if not isinstance(y_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'y' of 'dot_product' is not an ODE component")

        return Scalar(is_real_valued=False)


class _Len(Function):
    """len(state)`` returns the number of degrees of freedom in *x* """

    identifier = "<builtin>len"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, = self.resolve_args(arg_kinds)

        if not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return Scalar(is_real_valued=True)


class _IsNaN(Function):
    """isnan(x)`` returns True if there are any NaNs in *x*"""

    identifier = "<builtin>isnan"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, = self.resolve_args(arg_kinds)

        if not isinstance(x_kind, (NoneType, ODEComponent)):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return Boolean()


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
            (_Norm(), "{numpy}.linalg.norm({args})"),
            (_DotProduct(), "{numpy}.vdot({args})"),
            (_Len(), "len({args})"),
            (_IsNaN(), "{numpy}.isnan({args})"),
            ]:

        bfr = bfr.register(func)
        bfr = bfr.register_codegen(
            func.identifier,
            "python",
            _PythonBuiltinFunctionCodeGenerator(
                func, py_pattern))

    return bfr

base_function_registry = _make_bfr()

# }}}


# {{{ ODE RHS registration

class _ODERightHandSide(Function):
    default_dict = {}

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

    def get_result_kind(self, arg_kinds):
        arg_kinds = self.resolve_args(arg_kinds)

        if not isinstance(arg_kinds[0], (NoneType, Scalar)):
            raise TypeError("argument 't' of '%s' is not a scalar"
                    % self.identifier)

        for arg_name, arg_kind_passed, input_component_id in zip(
                self.arg_names[1:], arg_kinds[1:], self.input_component_ids):
            if arg_kind_passed is None:
                pass
            elif not (isinstance(arg_kind_passed, ODEComponent)
                    and arg_kind_passed.component_id == input_component_id):
                raise TypeError("argument '%s' of '%s' is not an ODE component "
                        "with component ID '%s'"
                        % (arg_name, self.identifier, input_component_id))

        return ODEComponent(self.component_id)


def register_ode_rhs(
        function_registry,
        component_id, identifier=None, input_component_ids=None,
        input_component_names=None):
    if identifier is None:
        identifier = component_id

    if input_component_ids is None:
        input_component_ids = (component_id,)

    return function_registry.register(
            _ODERightHandSide(
                identifier, component_id, input_component_ids,
                input_component_names=input_component_names))

# }}}

# vim: foldmethod=marker
