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

from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.dependency import DependencyMapper

import logging
import six.moves

logger = logging.getLogger(__name__)


class ExtendedDependencyMapper(DependencyMapper):
    """Extends DependencyMapper to handle values encountered in leap
    IR.
    """

    def map_foreign(self, expr):
        if expr is None or isinstance(expr, str):
            return frozenset()
        else:
            return super(ExtendedDependencyMapper, self).map_foreign(expr)


variable_mapper = ExtendedDependencyMapper(composite_leaves=False)


class EvaluationMapper(EvaluationMapperBase):

    def __init__(self, context, functions):
        """
        :arg context: a mapping from variable names to values
        :arg functions: a mapping from function names to functions
        """
        EvaluationMapperBase.__init__(self, context)
        self.functions = functions

    def map_generic_call(self, function_name, parameters, kw_parameters):
        if function_name in self.functions:
            function = self.functions[function_name]
        else:
            raise ValueError("Call to unknown function: " + str(function_name))
        evaluated_parameters = (self.rec(param) for param in parameters)
        evaluated_kw_parameters = dict(
                (param_id, self.rec(param))
                for param_id, param in six.iteritems(kw_parameters))
        return function(*evaluated_parameters, **evaluated_kw_parameters)

    def map_call(self, expr):
        return self.map_generic_call(expr.function.name, expr.parameters, {})

    def map_call_with_kwargs(self, expr):
        return self.map_generic_call(expr.function.name, expr.parameters,
                                     expr.kw_parameters)

# }}}

# vim: foldmethod=marker
