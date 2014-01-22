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


class FailStepException(Exception):
    pass


# {{{ states returned from run()

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


class NumpyInterpreter(object):
    """A :mod:`numpy`-targeting interpreter for the time integration language
    defined in :mod:`leap.vm.language`.
    """

    def __init__(self, code, rhs_map):
        self.code = code
        from leap.vm.language import ExecutionController
        self.exec_controller = ExecutionController(code)
        self.state = {}

        self.rhs_map = rhs_map

        from pymbolic.mapper.evaluator import EvaluationMapper
        self.eval_mapper = EvaluationMapper(self.state)

    def set_up(self, t_start, dt_start, state):
        """
        :arg state: a dictionary mapping state identifiers to their values
        """

        self.state["<t>"] = t_start
        self.state["<dt>"] = dt_start
        for key, val in state.iteritems():
            if key.startswith("<"):
                raise ValueError("state variables may not start with '<'")
            self.state[key] = val

    def initialize(self):
        for result in self.exec_controller(self.code.initialization_dep_on, self):
            yield result

    def run(self, t_end):
        while True:
            # {{{ adjust time step down at end of integration

            t = self.state["<t>"]
            dt = self.state["<t>"]
            if t+dt > t_end:
                self.state["<dt>"] = t_end - t

            # }}}

            try:
                try:
                    self.exec_controller.reset()
                    self.exec_controller.update_plan(self.code.step_dep_on)
                    for result in self.exec_controller(self):
                        yield result

                finally:
                    # discard non-permanent per-step state
                    for name in list(self.state.iterkeys()):
                        if not name.startswith("<p>") \
                                and name not in ["<t>", "<dt>"]:
                            del self.state[name]

            except FailStepException:
                yield StepFailed(t=self.state["<t>"])
                continue

            yield StepCompleted(t=self.state["<t>"])

    # {{{ execution methods

    def map_EvaluateRHS(self, insn):
        rhs = self.rhs_map[insn.rhs_id]
        t = self.eval_mapper(insn.t)

        for assignee, args in zip(insn.assignees, insn.rhs_arguments):
            self.state[assignee] = rhs(t, **dict(
                    (name, self.eval_mapper(expr))
                    for name, expr in args))

    def map_ReturnState(self, insn):
        yield StateComputed(
                t=self.eval_mapper(insn.time),
                time_id=insn.time_id,
                component_id=insn.component_id,
                state_component=self.eval_mapper(insn.expression))

    def map_EvaluateExpression(self, insn):
        self.state[insn.assignee] = self.eval_mapper(insn.expression)

    def map_Norm(self, insn):
        self.state[insn.assignee] = la.norm(
                self.eval_mapper(insn.expression), insn.p)

    def map_DotProduct(self, insn):
        self.state[insn.assignee] = np.vdot(
                self.eval_mapper(insn.expression_1),
                self.eval_mapper(insn.expression_2)
                )

    def map_Raise(self, insn):
        raise insn.error_condition(insn.error_message)

    def map_FailStep(self, insn):
        raise FailStepException()

    def map_If(self, insn):
        if self.eval_mapper(insn.condition):
            return insn.then_depends_on
        else:
            return insn.else_depends_on

    # }}}

# vim: fdm=marker
