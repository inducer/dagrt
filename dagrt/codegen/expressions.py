"""Code generation of expressions"""
from pymbolic.mapper.stringifier import (
        StringifyMapper, PREC_NONE, PREC_CALL, PREC_PRODUCT, PREC_LOGICAL_OR)
import numpy as np


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


# {{{ fortran

class FortranExpressionMapper(StringifyMapper):
    """Converts expressions to Fortran code."""

    def __init__(self, name_manager):
        """name_manager is a map from a variable name (as a string) to its
        representation (as a string).
        """
        super().__init__()
        self._name_manager = name_manager

    def map_constant(self, expr, enclosing_prec):
        if isinstance(expr, (complex, np.complex)):
            return "({}, {})".format(
                    self.rec(expr.real),
                    self.rec(expr.imag))
        elif isinstance(expr, bool):
            if expr:
                return ".true."
            else:
                return ".false."
        else:
            result = repr(expr).replace("e", "d")
            if "d" not in result:
                result = result+"d0"
            if expr < 0:
                result = "(%s)" % result
            return result

    def map_foreign(self, expr, enclosing_prec):
        if expr is None:
            raise NotImplementedError()
        elif isinstance(expr, str):
            return repr(expr)
        else:
            return super().map_foreign(
                    expr, enclosing_prec)

    TARGET_PREFIX = "<target>"

    def map_variable(self, expr, enclosing_prec):
        if expr.name.startswith(self.TARGET_PREFIX):
            return expr.name[len(self.TARGET_PREFIX):]
        else:
            return self._name_manager[expr.name]

    def map_lookup(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
                self.format("%s%%%s",
                    self.rec(expr.aggregate, PREC_CALL),
                    expr.name),
                enclosing_prec, PREC_CALL)

    def map_subscript(self, expr, enclosing_prec):
        if isinstance(expr.index, tuple):
            index_str = ", ".join(
                    "int(%s)" % self.rec(i, PREC_NONE)
                    for i in expr.index)
        else:
            index_str = "int(%s)" % self.rec(expr.index, PREC_NONE)

        return self.parenthesize_if_needed(
                self.format("%s(%s)",
                    self.rec(expr.aggregate, PREC_CALL),
                    index_str),
                enclosing_prec, PREC_CALL)

    def map_product(self, expr, enclosing_prec, *args, **kwargs):
        # This differs from the superclass only by adding spaces
        # around the operator, which provide an opportunity for
        # line breaking.
        return self.parenthesize_if_needed(
                self.join_rec(" * ", expr.children, PREC_PRODUCT, *args, **kwargs),
                enclosing_prec, PREC_PRODUCT)

    def map_logical_not(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_UNARY
        return self.parenthesize_if_needed(
                ".not. " + self.rec(expr.child, PREC_UNARY),
                enclosing_prec, PREC_UNARY)

    def map_logical_or(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_LOGICAL_OR
        return self.parenthesize_if_needed(
                self.join_rec(
                    " .or. ", expr.children, PREC_LOGICAL_OR),
                enclosing_prec, PREC_LOGICAL_OR)

    def map_logical_and(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_LOGICAL_AND
        return self.parenthesize_if_needed(
                self.join_rec(
                    " .and. ", expr.children, PREC_LOGICAL_AND),
                enclosing_prec, PREC_LOGICAL_AND)

# }}}


# {{{ python

class PythonExpressionMapper(StringifyMapper):
    """Converts expressions to Python code."""

    def __init__(self, name_manager, function_registry,
            numpy="numpy"):
        """name_manager is a map from a variable name (as a string) to its
        representation (as a string).

        numpy is the name of the numpy module.
        """
        super().__init__()
        self._name_manager = name_manager
        self._function_registry = function_registry
        self._numpy = numpy

    def map_constant(self, expr, *args):
        if isinstance(expr, (float, np.number)):
            if np.isinf(expr) or np.isnan(expr):
                return "float('" + repr(expr) + "')"
        return repr(expr)

    def map_foreign(self, expr, *args):
        if expr is None:
            return "None"
        elif isinstance(expr, str):
            return repr(expr)
        else:
            return super().map_foreign(expr, *args)

    def map_variable(self, expr, enclosing_prec):
        if expr.name.startswith("<func>"):
            return self._name_manager.name_function(expr.name)
        return self._name_manager[expr.name]

    def map_numpy_array(self, expr, *args):
        if len(expr.shape) > 1:
            raise ValueError(
                    "Representing multidimensional arrays is not supported")
        elements = [self.rec(element, *args) for element in expr]
        return "{numpy}.array([{elements}],dtype=\'object\')".format(
            numpy=self._numpy, elements=", ".join(elements))

    def map_generic_call(self, symbol, args, kwargs):
        arg_strs_dict = {}
        for i, arg in enumerate(args):
            arg_strs_dict[i] = self.rec(arg, PREC_NONE)
        for name, arg in kwargs.items():
            arg_strs_dict[name] = self.rec(arg, PREC_NONE)

        from dagrt.function_registry import FunctionNotFound
        try:
            codegen = self._function_registry.get_codegen(
                    symbol.name, "python")
        except FunctionNotFound:
            args_strs = [
                    self.rec(val, PREC_NONE)
                    for val in args
                    ] + [
                    "{name}={expr}".format(
                        name=name,
                        expr=self.rec(val, PREC_NONE))
                    for name, val in kwargs.items()]

            return "{rhs}({args})".format(
                    rhs=self._name_manager.name_function(symbol.name),
                    args=", ".join(args_strs))

        else:
            return codegen(self, arg_strs_dict)

    def map_call(self, expr, enclosing_prec):
        return self.map_generic_call(
                expr.function, expr.parameters, {})

    def map_call_with_kwargs(self, expr, enclosing_prec):
        return self.map_generic_call(
                expr.function, expr.parameters,
                expr.kw_parameters)

    def map_if(self, expr, enclosing_prec):
        from dagrt.expression import PREC_IFTHENELSE
        return self.parenthesize_if_needed(
            "{then} if {cond} else {else_}".format(
                # "1 if 0 if 1 else 2 else 3" is not valid Python.
                # So force parens by using an artificially higher precedence.
                then=self.rec(expr.then, PREC_LOGICAL_OR),
                cond=self.rec(expr.condition, PREC_LOGICAL_OR),
                else_=self.rec(expr.else_, PREC_LOGICAL_OR)),
            enclosing_prec, PREC_IFTHENELSE)

# }}}

# vim: foldmethod=marker
