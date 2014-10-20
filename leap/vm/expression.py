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
        PREC_NONE,
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


class RHSEvaluation(LeapExpression):
    """
    .. attribute:: rhs_id

        Identifier of the right hand side to be evaluated. Typically a number
        or a string.

    .. attribute:: t

        A :mod:`pymbolic` expression for the time at which the right hand side
        is to be evaluated. Time is implicitly the first argument to each
        expression.

    .. attribute:: arguments

        A tuple of tuples. The outer tuple is the vectorization list, where each
        entry corresponds to one of the entries of :attr:`assignees`. The inner
        lists corresponds to arguments being passed to the right-hand side
        (identified by :attr:`component_id`) being invoked. These are tuples
        ``(arg_name, expression)``, where *expression* is a :mod:`pymbolic`
        expression.
    """
    def __init__(self, rhs_id, t, arguments):
        self.rhs_id = rhs_id
        self.t = t
        self.arguments = arguments

    def __getinitargs__(self):
        return (self.t, self.arguments)

    mapper_method = six.moves.intern("map_rhs_evaluation")


class Norm(LeapExpression):
    """
    .. attribute:: argument
    .. attribute:: p
    """
    def __init__(self, expression, p):
        self.expression = expression
        self.p = p

    def __getinitargs__(self):
        return (self.expression, self.p)

    mapper_method = six.moves.intern("map_norm")


class DotProduct(LeapExpression):
    """
    .. attribute:: argument_1

        The complex conjugate of this argument is taken before computing the
        dot product, if applicable.

    .. attribute:: argument_2
    """
    def __init__(self, argument_1, argument_2):
        self.argument_1 = argument_1
        self.argument_2 = argument_2

    def __getinitargs__(self):
        return (self.argument_1, self.argument_2)

    mapper_method = six.moves.intern("map_dot_product")

# }}}


# {{{ mappers

class StringifyMapper(StringifyMapperBase):
    def map_rhs_evaluation(self, expr, enclosing_prec):
        return "rhs:%s(%s)" % (
                expr.rhs_id,
                ", ".join(
                    self.rec(arg, PREC_NONE)
                    for var, arg in expr.arguments))

    def map_norm(self, expr, enclosing_prec):
        return "||%s||_%s" % (
                self.rec(expr.expression, PREC_NONE),
                self.rec(expr.p, PREC_NONE))

    def map_dot_product(self, expr, enclosing_prec):
        return "(%s, %s)" % (
                self.rec(expr.expression_1, PREC_NONE),
                self.rec(expr.expression_2, PREC_NONE))


class CombineMapper(CombineMapperBase):
    def map_rhs_evaluation(self, expr):
        return self.combine(
                [self.rec(expr.t)]
                + [self.rec(val) for name, val in expr.arguments])

    def map_norm(self, expr):
        return self.rec(expr.expression)

    def map_dot_product(self, expr):
        return self.combine([
            self.rec(expr.expression_1),
            self.rec(expr.expression_2),
            ])


class DependencyMapper(DependencyMapperBase, CombineMapper):
    pass


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
    def __init__(self, context, functions, rhs_map):
        """
        :arg context: a mapping from variable names to values
        """
        EvaluationMapperBase.__init__(self, context)
        self.functions = functions
        self.rhs_map = rhs_map

    def map_call(self, expr):
        func = self.functions[expr.function.name]
        return func(*[self.rec(par) for par in expr.parameters])

    def map_rhs_evaluation(self, expr):
        rhs = self.rhs_map[expr.rhs_id]
        t = self.rec(expr.t)
        return rhs(t, **dict(
            (name, self.rec(expr))
            for name, expr in expr.arguments))

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
