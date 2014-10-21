"""Code generation of expressions"""

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

from pymbolic.mapper.stringifier import (
        StringifyMapper, PREC_NONE)
from pytools import DictionaryWithDefault
import numpy as np
from leap.vm.expression import Norm


# {{{ fortran

class FortranExpressionMapper(StringifyMapper):
    """Converts expressions to Fortran code."""

    def __init__(self, name_manager):
        """name_manager is a map from a variable name (as a string) to its
        representation (as a string).
        """
        super(FortranExpressionMapper, self).__init__(repr)
        self._name_manager = name_manager

    def map_foreign(self, expr, *args):
        if expr is None:
            raise NotImplementedError()
        elif isinstance(expr, str):
            return repr(expr)
        else:
            return super(FortranExpressionMapper, self).map_foreign(expr, *args)

    def map_variable(self, expr, enclosing_prec):
        return self._name_manager[expr.name]

    def map_numpy_array(self, expr, *args):
        if len(expr.shape) > 1:
            raise ValueError('Representing multidimensional arrays is ' +
                             'not supported')
        elements = [self.rec(element, *args) for element in expr]
        return '{numpy}.array([{elements}],dtype=\'object\')'.format(
            numpy=self._numpy, elements=', '.join(elements))

# }}}


# {{{ python

class PythonExpressionMapper(StringifyMapper):
    """Converts expressions to Python code."""

    def __init__(self, name_manager, numpy='numpy'):
        """name_manager is a map from a variable name (as a string) to its
        representation (as a string).

        numpy is the name of the numpy module.
        """
        super(PythonExpressionMapper, self).__init__(repr)
        self._name_manager = name_manager
        self._numpy = numpy

    def map_foreign(self, expr, *args):
        if expr is None:
            return 'None'
        elif isinstance(expr, str):
            return repr(expr)
        else:
            return super(PythonExpressionMapper, self).map_foreign(expr, *args)

    def map_variable(self, expr, enclosing_prec):
        return self._name_manager[expr.name]

    def map_numpy_array(self, expr, *args):
        if len(expr.shape) > 1:
            raise ValueError('Representing multidimensional arrays is ' +
                             'not supported')
        elements = [self.rec(element, *args) for element in expr]
        return '{numpy}.array([{elements}],dtype=\'object\')'.format(
            numpy=self._numpy, elements=', '.join(elements))

    def map_generic_call(self, symbol, args, kwargs):
        if isinstance(symbol, Norm):
            assert len(args) == 1
            assert kwargs == {}
            order = symbol.p
            if isinstance(order, (float, np.number)) and np.isinf(order):
                order_str = "float('inf')"
            else:
                order_str = self.rec(order, PREC_NONE)
            return '{numpy}.linalg.norm({expr}, ord={ord})'.format(
                numpy=self._numpy, expr=self.rec(args[0], PREC_NONE),
                ord=order_str)

        args_strs = [
                self.rec(val, PREC_NONE)
                for val in args
                ] + [
                '{name}={expr}'.format(
                    name=name,
                    expr=self.rec(val, PREC_NONE))
                for name, val in kwargs]

        return 'self._functions.{rhs}({args})'.format(
                rhs=self._name_manager.name_function(symbol),
                args=", ".join(args_strs))

    def map_call(self, expr, enclosing_prec):
        return self.map_generic_call(
                expr.function, expr.parameters, {})

    def map_call_with_kwargs(self, expr, enclosing_prec):
        return self.map_generic_call(
                expr.function, expr.parameters,
                expr.kw_parameters.items())


string_mapper = PythonExpressionMapper(DictionaryWithDefault(lambda x: x))

# }}}

# vim: foldmethod=marker
