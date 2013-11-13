from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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


from pytools import Record


# {{{ instructions

# Variables whose names start with the pattern <letters> are special.
# The following variable tags are supported:
#
# <p> : This variable contains persistent state. (that survives from ...?)
# <dt> : time step
# <t> : base time of current time step

class Instruction(Record):
    """
    .. attribute:: id
    .. attribute:: depends_on
    """


class EvaluateRHS(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: out_component
    .. attribute:: in_component
    .. attribute:: evaluation_id
    .. attribute:: t
    .. attribute:: states
    """


class SolveRHS(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: out_component
    .. attribute:: in_component
    .. attribute:: evaluation_id
    .. attribute:: t
    .. attribute:: states
    .. attribute:: solve_component
    .. attribute:: rhs
    """


class ReturnState(Instruction):
    """
    .. attribute:: time_id
    .. attribute:: time
    .. attribute:: component
    .. attribute:: variable
    """


class Assign(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: variable
    """


class Norm(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: expression
    .. attribute:: p
    """


class DotProduct(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: expression_1
    .. attribute:: expression_2
    """


class Raise(Instruction):
    """
    .. attribute:: error_condition
    .. attribute:: error_message
    """


class FailStep(Instruction):
    """
    """


class If(Instruction):
    """
    .. attribute:: condition
    .. attribute:: then_depends_on

        a set of ids that the instruction depends on if :attr:`condition_expr`
        evaluates to True

    .. attribute:: else_depends_on

        a set of ids that the instruction depends on if :attr:`condition`
        evaluates to False
    """

# }}}

# light cone optimizations?


# {{{ method

class Method(Record):
    """
    .. atttribute:: initialization_dep_on
    .. atttribute:: step_dep_on
    .. atttribute:: steady_state_step_dep_on
    """

# }}}

# vim: fdm=marker
