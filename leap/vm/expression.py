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

from pymbolic.primitives import Expression
from pymbolic.mapper import (  # noqa
        IdentityMapper as IdentityMapperBase,
        CombineMapper as CombineMapperBase,
        WalkMapper as WalkMapperBase,
        )
from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.differentiator import DifferentiationMapper as \
    DifferentiationMapperBase

from pymbolic.mapper.dependency import DependencyMapper as DependencyMapperBase

from pymbolic.mapper.stringifier import (
        StringifyMapper as StringifyMapperBase)

import logging
import six.moves

import numpy as np
import numpy.linalg as la

logger = logging.getLogger(__name__)


# {{{ symbol nodes

class LeapExpression(Expression):
    def stringifier(self):
        return StringifyMapper


class Norm(LeapExpression):
    """Function symbol for norm evaluation.

    .. attribute:: p
    """
    def __init__(self, expression, p):
        self.p = p

    def __getinitargs__(self):
        return (self.expression, self.p)

    mapper_method = six.moves.intern("map_norm_symbol")


class DotProduct(LeapExpression):
    """Function symbol for dot product evaluation."""

    mapper_method = six.moves.intern("map_dot_product_symbol")

# }}}


# {{{ mappers

class StringifyMapper(StringifyMapperBase):
    def map_norm_symbol(self, expr, enclosing_prec):
        return "norm[%p]" % self.rec(expr.p)

    def map_dot_product_symbol(self, expr, enclosing_prec):
        return "dot"


class CombineMapper(CombineMapperBase):
    pass


class DependencyMapper(DependencyMapperBase, CombineMapper):
    def map_norm(self, expr):
        return frozenset()

    map_dot_product = map_norm


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
    def __init__(self, context, execution_backend):
        """
        :arg context: a mapping from variable names to values
        """
        EvaluationMapperBase.__init__(self, context)
        self.execution_backend = execution_backend

    def map_call(self, expr):
        func = self.execution_backend.resolve(expr.function)
        return func(
                *[self.rec(par) for par in expr.parameters],
                **dict(
                    (name, self.rec(par))
                    for name, par in expr.kw_parameters))

    def map_norm(self, expr):
        return la.norm(
                self.rec(expr.expression),
                self.rec(expr.p))

    def map_dot_product(self, expr):
        return np.vdot(
                self.rec(expr.expression_1),
                self.rec(expr.expression_2))


class DifferentiationMapperWithContext(DifferentiationMapperBase):

    def __init__(self, variable, functions, context):
        DifferentiationMapperBase.__init__(self, variable, None)
        self.context = context
        self.functions = functions

    def map_call(self, expr):
        raise NotImplementedError

    def map_variable(self, expr):
        return self.context[expr.name] if expr.name in self.context else \
            DifferentiationMapperBase.map_variable(self, expr)

# }}}

# vim: foldmethod=marker
