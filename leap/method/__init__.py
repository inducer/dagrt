"""Time integration methods."""

from __future__ import division

__copyright__ = "Copyright (C) 2007-2013 Andreas Kloeckner"

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


class Method(object):

    def generate(self, *solver_hooks):
        """
        Generate a method description.

        :arg solver_hooks: A list of callbacks that generate expressions
        for calling user-supplied implicit solvers

        :return: A `TimeIntegratorCode` instance
        """
        raise NotImplementedError()

    def implicit_expression(self, expression_tag=None):
        """
        Return a template that expressions in `class`:AssignSolved
        instances will follow.

        :arg expression_tag: A name for the expression, if multiple
        expressions are present in the generated code.

        :return: A tuple consisting of :mod:`pymbolic` expressions and
        the names of the free variables in the expressions.
        """
        raise NotImplementedError()


# {{{ diagnostics

class TimeStepUnderflow(RuntimeError):
    pass

# }}}
