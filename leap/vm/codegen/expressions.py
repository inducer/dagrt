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

from pymbolic.mapper.stringifier import StringifyMapper
from pytools import DictionaryWithDefault


class PythonExpressionMapper(StringifyMapper):
    """Converts expressions to Python code."""

    def __init__(self, variable_names, numpy='numpy'):
        super(PythonExpressionMapper, self).__init__(repr)
        self.variable_names = variable_names
        self.numpy = numpy

    def map_foreign(self, expr, *args):
        if expr is None:
            return 'None'
        elif isinstance(expr, str):
            return repr(expr)
        else:
            return super(PythonExpressionMapper, self).map_foreign(expr, *args)

    def map_variable(self, expr, enclosing_prec):
        return self.variable_names[expr.name]

    def map_numpy_array(self, expr, *args):
        if len(expr.shape) > 1:
            raise ValueError('Representing multidimensional arrays is ' +
                             'not supported')
        elements = [self.rec(element, *args) for element in expr]
        return '%s.array([%s],dtype=\'object\')' % (self.numpy,
                                                    ', '.join(elements))


string_mapper = PythonExpressionMapper(DictionaryWithDefault(lambda x: x))
