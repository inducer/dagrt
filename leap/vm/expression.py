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

from pytools import RecordWithoutPickling, memoize_method
from pymbolic.primitives import Expression

from pymbolic.mapper.stringifier import (
        StringifyMapper as StringifyMapperBase)

from leap.vm.utils import get_variables

import logging
import six
import six.moves

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

    mapper_method = "map_rhs_evaluation"


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

    mapper_method = "map_norm"


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

    mapper_method = "map_dot_product"

# }}}


# {{{ mappers

class StringifyMapper(StringifyMapperBase):
    def map_norm(self, expr):
        return "||%s||_%s" % (expr.argument, self.p)


# }}}

# vim: foldmethod=marker
