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

from leap.vm.expression import EvaluationMapper
from collections import namedtuple
import numpy as np

import six


class FailStepException(Exception):
    pass


class NumpyInterpreter(object):
    """A :mod:`numpy`-targeting interpreter for the time integration language
    defined in :mod:`leap.vm.language`.

    .. automethod:: set_up
    .. automethod:: initialize
    .. automethod:: run
    """

    # {{{ events returned from run()

    class StateComputed(namedtuple("StateComputed",
              ["t", "time_id", "component_id", "state_component"])):
        """
        .. attribute:: t
        .. attribute:: time_id
        .. attribute:: component_id

            Identifier of the state component being returned.

        .. attribute:: state_component
        """

    class StepCompleted(namedtuple("StepCompleted", ["t"])):
        """
        .. attribute:: t

            Floating point number.
        """

    class StepFailed(namedtuple("StepFailed", ["t"])):
        """
        .. attribute:: t

            Floating point number.
        """
    # }}}

    def __init__(self, code, function_map):
        """
        :arg code: an instance of :class:`leap.vm.TimeIntegratorCode`
        :arg function_map: a mapping from function identifiers to functions
        """
        self.code = code
        from leap.vm.language import ExecutionController
        self.exec_controller = ExecutionController(code)
        self.context = {}
        self.next_state = self.code.initial_state
        builtins = {
                "<builtin>len": len,
                "<builtin>isnan": np.isnan,
                "<builtin>norm": np.linalg.norm,
                "<builtin>dot_product": np.vdot
                }

        # Ensure none of the names in the function map conflict with the
        # builtins.
        assert not set(builtins) & set(function_map)

        self.functions = dict(builtins, **function_map)

        self.eval_mapper = EvaluationMapper(self.context, self.functions)

    def set_up(self, t_start, dt_start, context):
        """
        :arg context: a dictionary mapping identifiers to their values
        """

        self.context["<t>"] = t_start
        self.context["<dt>"] = dt_start
        for key, val in six.iteritems(context):
            if key.startswith("<"):
                raise ValueError("state variables may not start with '<'")
            self.context["<state>"+key] = val

    def initialize(self):
        self.exec_controller.reset()
        cur_state = self.code.states[self.next_state]
        self.next_state = cur_state.next_state
        self.exec_controller.update_plan(cur_state.depends_on)
        for event in self.exec_controller(self):
            pass

    def run(self, t_end):
        """Generates :ref:`numpy-exec-events`."""

        last_step = False
        while True:
            # {{{ adjust time step down at end of integration

            t = self.context["<t>"]
            dt = self.context["<dt>"]

            if t+dt >= t_end:
                assert t <= t_end
                self.context["<dt>"] = t_end - t
                last_step = True

            # }}}

            try:
                try:
                    self.exec_controller.reset()
                    cur_state = self.code.states[self.next_state]
                    self.next_state = cur_state.next_state
                    self.exec_controller.update_plan(cur_state.depends_on)
                    for event in self.exec_controller(self):
                        yield event

                finally:
                    # discard non-permanent per-step state
                    for name in list(six.iterkeys(self.context)):
                        if (
                                not name.startswith("<state>")
                                and not name.startswith("<p>")
                                and name not in ["<t>", "<dt>"]):
                            del self.context[name]

            except FailStepException:
                yield self.StepFailed(t=self.context["<t>"])
                continue

            yield self.StepCompleted(t=self.context["<t>"])

            if last_step:
                break

    def register_function(self, name, f):
        if name in self.functions:
            raise ValueError("function '%s' already regsitered" % name)

        self.functions[name] = f

    # {{{ execution methods

    def exec_YieldState(self, insn):
        return self.StateComputed(
                    t=self.eval_mapper(insn.time),
                    time_id=insn.time_id,
                    component_id=insn.component_id,
                    state_component=self.eval_mapper(insn.expression)), []

    def exec_AssignExpression(self, insn):
        self.context[insn.assignee] = self.eval_mapper(insn.expression)

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
