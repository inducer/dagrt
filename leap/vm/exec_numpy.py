from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import numpy as np
import numpy.linalg as la

from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase


class FailStepException(Exception):
    pass


# {{{ events returned from run()

class StateComputed(Record):
    """
    .. attribute:: t
    .. attribute:: time_id
    .. attribute:: component_id

        Identifier of the state component being returned.

    .. attribute:: state_component
    """


class StepCompleted(Record):
    """
    .. attribute:: t

        Floating point number.

    .. attribute:: number

        Integer, initial state is 0.
    """


class StepFailed(Record):
    """
    .. attribute:: t

        Floating point number.

    .. attribute:: number

        Integer, initial state is 0.
    """

# }}}


class EvaluationMapper(EvaluationMapperBase):
    def __init__(self, context, functions):
        """
        :arg context: a mapping from variable names to values
        """
        EvaluationMapperBase.__init__(self, context)
        self.functions = functions

    def map_call(self, expr):
        func = self.functions[expr.function.name]
        return func(*[self.rec(par) for par in expr.parameters])


class NumpyInterpreter(object):
    """A :mod:`numpy`-targeting interpreter for the time integration language
    defined in :mod:`leap.vm.language`.

    .. automethod:: set_up
    .. automethod:: initialize
    .. automethod:: run
    """

    def __init__(self, code, rhs_map):
        """
        :arg code: an instance of :class:`leap.vm.TimeIntegratorCode`
        :arg rhs_map: a mapping from component ids to right-hand-side
            functions
        """
        self.code = code
        from leap.vm.language import ExecutionController
        self.exec_controller = ExecutionController(code)
        self.state = {}
        self.functions = {
                "len": len,
                "isnan": np.isnan,
                }

        self.rhs_map = rhs_map

        self.eval_mapper = EvaluationMapper(self.state, self.functions)

    def set_up(self, t_start, dt_start, state):
        """
        :arg state: a dictionary mapping state identifiers to their values
        """

        self.state["<t>"] = t_start
        self.state["<dt>"] = dt_start
        for key, val in state.iteritems():
            if key.startswith("<"):
                raise ValueError("state variables may not start with '<'")
            self.state["<state>"+key] = val

    def initialize(self):
        self.exec_controller.reset()
        self.exec_controller.update_plan(self.code.initialization_dep_on)
        for event in self.exec_controller(self):
            pass

    def run(self, t_end):
        """Generates :ref:`numpy-exec-events`."""

        last_step = False
        while True:
            # {{{ adjust time step down at end of integration

            t = self.state["<t>"]
            dt = self.state["<dt>"]

            if t+dt >= t_end:
                assert t <= t_end
                self.state["<dt>"] = t_end - t
                last_step = True

            # }}}

            try:
                try:
                    self.exec_controller.reset()
                    self.exec_controller.update_plan(self.code.step_dep_on)
                    for event in self.exec_controller(self):
                        yield event

                finally:
                    # discard non-permanent per-step state
                    for name in list(self.state.iterkeys()):
                        if (
                                not name.startswith("<state>")
                                and not name.startswith("<p>")
                                and name not in ["<t>", "<dt>"]):
                            del self.state[name]

            except FailStepException:
                yield StepFailed(t=self.state["<t>"])
                continue

            yield StepCompleted(t=self.state["<t>"])

            if last_step:
                break

    def register_function(self, name, f):
        if name in self.functions:
            raise ValueError("function '%s' already regsitered" % name)

        self.functions[name] = f

    # {{{ execution methods

    def exec_AssignRHS(self, insn):
        rhs = self.rhs_map[insn.component_id]
        t = self.eval_mapper(insn.t)

        for assignee, args in zip(insn.assignees, insn.rhs_arguments):
            self.state[assignee] = rhs(t, **dict(
                    (name, self.eval_mapper(expr))
                    for name, expr in args))

    def exec_ReturnState(self, insn):
        return StateComputed(
                    t=self.eval_mapper(insn.time),
                    time_id=insn.time_id,
                    component_id=insn.component_id,
                    state_component=self.eval_mapper(insn.expression)), []

    def exec_AssignExpression(self, insn):
        self.state[insn.assignee] = self.eval_mapper(insn.expression)

    def exec_AssignNorm(self, insn):
        self.state[insn.assignee] = la.norm(
                self.eval_mapper(insn.expression), insn.p)

    def exec_AssignDotProduct(self, insn):
        self.state[insn.assignee] = np.vdot(
                self.eval_mapper(insn.expression_1),
                self.eval_mapper(insn.expression_2)
                )

    def exec_Raise(self, insn):
        raise insn.error_condition(insn.error_message)

    def exec_FailStep(self, insn):
        raise FailStepException()

    def exec_If(self, insn):
        if self.eval_mapper(insn.condition):
            return None, insn.then_depends_on
        else:
            return None, insn.else_depends_on

    # }}}

# vim: fdm=marker
