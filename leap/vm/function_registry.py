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

import six
from leap.vm.utils import TODO
from pytools import RecordWithoutPickling
from leap.vm.codegen.data import ODEComponent, Boolean, Scalar


# {{{ function

class Function(RecordWithoutPickling):
    """
    .. attribute:: name
    .. attribute:: arg_names
    .. attribute:: default_dict
    """

    def __init__(self, _language_to_codegen=None):
        if _language_to_codegen is None:
            _language_to_codegen = {}

        super(Function, self).__init__(
                _language_to_codegen=_language_to_codegen)

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
        new_language_to_codegen = self._language_to_codegen.copy()

        if language in new_language_to_codegen:
            raise ValueError("a code generator for function '%s' in "
                    "language '%s' is already known"
                    % (self.name, language))

        new_language_to_codegen[language] = codegen_function
        return self.copy(_language_to_codegen=new_language_to_codegen)

    def get_codegen(self, language):
        try:
            return self._language_to_codegen[language]
        except KeyError:
            return ValueError(
                    "'%s' has no code generator for language '%s'"
                    % (self.name, language))

    def resolve_args(self, arg_dict):
        from leap.vm.utils import resolve_args
        return resolve_args(self.arg_names, self.default_dict, arg_dict)

# }}}


# {{{ function registry

class FunctionRegistry(RecordWithoutPickling):
    def __init__(self, _id_to_function=None):
        if _id_to_function is None:
            _id_to_function = {}

        super(FunctionRegistry, self).__init__(self,
                _id_to_function=_id_to_function)

    def register(self, function):
        """Return a copy of *self* with *function* registered."""

        if function.identifier in self._id_to_function:
            raise ValueError("function '%s' is already registered"
                    % function.identifier)

        new_id_to_function = self._id_to_function.copy()
        new_id_to_function[function.identifier] = function
        return self.copy(_id_to_function=new_id_to_function)

    def register_codegen(self, function_id, language, codegen):
        """Register a code generation helper object for target *language*
        for the function with identifier *function_id*.

        :arg codegen: an object obeying an interface suitable for code
            generation for *language*. This interface depends on the code
            generator being used.
        """
        func = (self._id_to_function[function_id]
                .register_codegen(language, codegen))

        new_id_to_function = self._id_to_function.copy()
        new_id_to_function[function_id] = func
        return self.copy(_id_to_function=new_id_to_function)

    def get_codegen(self, function_id, language):
        try:
            func = self._id_to_function[function_id]
        except KeyError:
            raise NameError(
                    "unknown function: '%s'"
                    % function_id)

        return func.get_codegen(language)

    def register_right_hand_sides(self, rhs_list):
        raise TODO()

    def copy(self):
        return dict(
                (identifier, func.copy())
                for identifier, func in six.iteritems(self._id_to_function))

# }}}


# {{{ built-in functions

class _Norm(Function):
    """norm(x, ord)`` returns the *ord*-norm of *x*."""

    name = "<builtin>norm"
    arg_names = ("x", "ord")
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, ord_kind = self.resolve_args(arg_kinds)

        if not isinstance(ord_kind, Scalar):
            raise TypeError("argument 'ord' of 'norm' is not a scalar")
        if not isinstance(x_kind, ODEComponent):
            raise TypeError("argument 'x' of 'norm' is not an ODE component")

        return Scalar(is_real_valued=True)


class _DotProduct(Function):
    """dot_product(x, y)`` return the dot product of *x* and *y*. The
    complex conjugate of *x* is taken first, if applicable.
    """

    name = "<builtin>dot_product"
    arg_names = ("x", "y")
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, y_kind = self.resolve_args(arg_kinds)

        if not isinstance(x_kind, ODEComponent):
            raise TypeError("argument 'x' of 'dot_product' is not an ODE component")
        if not isinstance(y_kind, ODEComponent):
            raise TypeError("argument 'y' of 'dot_product' is not an ODE component")

        return Scalar(is_real_valued=False)


class _Len(Function):
    """len(state)`` returns the number of degrees of freedom in *x* """

    name = "<builtin>len"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, = self.resolve_args(arg_kinds)

        if not isinstance(x_kind, ODEComponent):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return Scalar(is_real_valued=True)


class _IsNaN(Function):
    """isnan(x)`` returns True if there are any NaNs in *x*"""

    name = "<builtin>isnan"
    arg_names = ("x",)
    default_dict = {}

    def get_result_kind(self, arg_kinds):
        x_kind, = self.resolve_args(arg_kinds)

        if not isinstance(x_kind, ODEComponent):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return Boolean()


class _PythonBuiltinFunctionCodeGenerator(object):
    def __init__(self, function, pattern):
        self.function = function
        self.pattern = pattern

    def __call__(self, expr_mapper, arg_strs_dict):
        args = self.function.resolve_args(arg_strs_dict)
        return self.pattern.format(
            numpy=self.expr_mapper.numpy,
            args=", ".join(args))


base_function_registry = FunctionRegistry()

for func, py_pattern in [
        (_Norm(), "{numpy}.linalg.norm({args})"),
        (_DotProduct(), "{numpy}.vdot({args})"),
        (_Len(), "len({args})"),
        (_IsNaN(), "{numpy}.isnan({args})"),
        ]:

    base_function_registry.register(func)
    base_function_registry.register_codegen(
        func.name,
        _PythonBuiltinFunctionCodeGenerator(
            func, py_pattern))

# }}}


# {{{ ODE RHS registration

class _ODERightHandSide(Function):
    default_dict = {}

    def __init__(self, name, component_id, input_component_ids):
        self.name = name
        self.component_id = component_id
        self.arg_names = input_component_ids

    def get_result_kind(self, arg_kinds):
        arg_kinds, = self.resolve_args(arg_kinds)

        for arg_kind_passed, arg_kind_needed in zip(
                arg_kinds, self.
        if not isinstance(x_kind, ODEComponent):
            raise TypeError("argument 'x' of 'len' is not an ODE component")

        return Boolean()
def register_ode_rhs(name, component_id=None, input_component_ids=None):

    pass

# }}}

# vim: foldmethod=marker
