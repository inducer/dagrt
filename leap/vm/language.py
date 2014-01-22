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

from pytools import Record, memoize_method


# {{{ instructions

__doc__ = """
Identifier conventions
----------------------

Identifiers whose names start with the pattern <letters> are special.  The
following special variable names are supported:

<p>NAME : This variable contains persistent state.
  (that survives from one step to the next)

<dt> : time step
  Its value at the beginning of a step indicates the step size to be used. If
  a time step of this size cannot be completed, FailStep must be issued.

<t> : base time of current time step.
  The integrator code is responsible for incrementing <t> at the end of a
  successful step.

<state>NAME : state identifier under user control

<func>NAME : A user-defined function.

The latter two serve to separate the name space used by the method from that
under the control of the user.
"""


class Instruction(Record):
    """
    .. attribute:: id

        A string, a unique identifier for this instruction.

    .. attribute:: depends_on

        A :class:`frozenset` of instruction ids that are reuqired to be
        executed within this execution context before this instruction can be
        executed.
    """


class EvaluateRHS(Instruction):
    """
    .. attribute:: assignees

        A tuple of strings, the names of the variable being assigned to.

    .. attribute:: rhs_id

        Identifier of the right hand side to be evaluated. Typically a number
        or a string.

    .. attribute:: t

        A :mod:`pymbolic` expression for the time at which the right hand side
        is to be evaluated. Time is implicitly the first argument to each
        expression.

    .. attribute:: rhs_arguments

        A tuple of tuples. The outer tuple is the vectorization list, where each
        entry corresponds to one of the entries of :attr:`assignees`. The inner
        lists corresponds to arguments being passed to the right-hand side
        (identified by :attr:`rhs_id`) being invoked. These are tuples
        ``(arg_name, expression)``, where *expression* is a :mod:`pymbolic`
        expression.
    """


class SolveRHS(Instruction):
    # FIXME: We should allow vectorization over multiple inputs/outputs
    # Pure vectorization is not enough here. We may want some amount
    # of tensor product expression flexibility.

    # FIXME This needs some thought.
    """
    .. attribute:: assignee

        A string, the name of the variable being assigned to.

    .. attribute:: rhs_id
    .. attribute:: t
    .. attribute:: states
    .. attribute:: solve_component
    .. attribute:: rhs
    """


class ReturnState(Instruction):
    """
    .. attribute:: time_id
    .. attribute:: time

        A :mod:`pymbolic` expression representing the time at which the
        state returned is valid.

    .. attribute:: component_id
    .. attribute:: expression
    """


class EvaluateExpression(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: expression
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

        The complex conjugate of this argument is taken before computing the
        dot product, if applicable.

    .. attribute:: expression_2
    """


class Raise(Instruction):
    """
    .. attribute:: error_condition

        An exception type to be raised.

    .. attribute:: error_message

        The error message to go along with that exception type.
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


# {{{ time integrator code

class TimeIntegratorCode(Record):
    """
    .. atttribute:: initialization_dep_on

        List of instruction ids (not including their recursive dependencies) to
        be executed for initialization of the time integrator.

    .. atttribute:: step_dep_on

        List of instruction ids (not including their recursive dependencies) to
        be executed for one time step.
    """

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn)
                for insn in self.instructions)


class ExecutionController(object):
    def __init__(self, code):
        self.code = code

        self.plan = []
        self.executed_ids = set()
        self.plan_id_set = set()

    def reset(self):
        self.execute_ids.clear()

    def update_plan(self, execute_ids):
        """Update the plan with the minimal list of instruction ids to execute
        so that the instruction IDs in execute_ids will be executed before any
        others and such that all their dependencies are satisfied.
        """

        early_plan_id_set = set()
        early_plan = []

        id_to_insn = self.code.id_to_insn

        def add_with_deps(insn):
            insn_id = insn.id
            if insn_id in early_plan_id_set:
                return

            for dep_id in insn.depends_on:
                add_with_deps(id_to_insn[dep_id])

            if insn_id in self.plan_id_set and insn_id not in self.executed_ids:
                self.plan_id_set.remove(insn_id)
                self.plan.remove(insn_id)

            assert insn_id not in early_plan_id_set
            early_plan.append(insn_id)
            early_plan_id_set.add(insn_id)

        for insn_id in execute_ids:
            add_with_deps(id_to_insn[insn_id])

        self.plan = early_plan + self.plan
        self.plan_id_set.update(early_plan_id_set)

    def __call__(self, target):
        id_to_insn = self.code.id_to_insn

        while self.plan:
            insn_id = self.plan.pop(0)
            self.executed_ids.append(insn_id)

            insn = id_to_insn[insn_id]

            new_deps = getattr(target, insn.exec_method)(insn)
            if new_deps is not None:
                self.update_plan(new_deps)

# }}}


def infer_single_writer_dependencies(insns):
    """Return a new set of :class:`Instruction` instances with reader dependencies
    added where a single instruction writes one variable.
    """

    raise NotImplementedError

# vim: fdm=marker
