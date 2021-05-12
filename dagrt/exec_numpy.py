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

from collections import namedtuple
from dagrt.expression import EvaluationMapper


__doc__ = """
.. autoclass:: StateComputed
.. autoclass:: StepCompleted
.. autoclass:: StepFailed

.. autoexception:: FailStepException
.. autoclass:: TransitionEvent
.. autoclass:: NumpyInterpreter
"""


class FailStepException(Exception):
    pass


class TransitionEvent(Exception):

    def __init__(self, next_phase):
        self.next_phase = next_phase


# {{{ events returned from NumpyInterpreter.run()

class StateComputed(namedtuple("StateComputed",
          ["t", "time_id", "component_id", "state_component"])):
    """
    .. attribute:: t
    .. attribute:: time_id
    .. attribute:: component_id

        Identifier of the state component being returned.

    .. attribute:: state_component
    """


class StepCompleted(
        namedtuple("StepCompleted",
            ["dt", "t", "current_state", "next_phase"])):
    """
    .. attribute:: dt

        Size of next time step.

    .. attribute:: t

        Approximate integrator time at end of step.

    .. attribute:: current_state
    .. attribute:: next_phase
    """


class StepFailed(namedtuple("StepFailed", ["t"])):
    """
    .. attribute:: t

        Floating point number.
    """
# }}}


# {{{ interpreter

class NumpyInterpreter:
    """A :mod:`numpy`-targeting interpreter for the time integration language
    defined in :mod:`dagrt.language`.

    Implements

    .. attribute:: next_phase

    .. attribute:: StateComputed
    .. attribute:: StepCompleted
    .. attribute:: StepFailed

    .. automethod:: set_up
    .. automethod:: run
    .. automethod:: run_single_step
    """

    StateComputed = StateComputed
    StepCompleted = StepCompleted
    StepFailed = StepFailed

    def __init__(self, code, function_map):
        """
        :arg code: an instance of :class:`dagrt.language.DAGCode`
        :arg function_map: a mapping from function identifiers to functions
        """
        self.code = code
        from dagrt.language import ExecutionController
        self.exec_controller = ExecutionController(code)
        self.context = {}
        self.next_phase = self.code.initial_phase

        from dagrt.builtins_python import builtins

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
        for key, val in context.items():
            if key.startswith("<"):
                raise ValueError("state variables may not start with '<'")
            self.context["<state>"+key] = val

    def run(self, t_end=None, max_steps=None):
        """Generates events."""

        n_steps = 0
        while True:
            if t_end is not None and self.context["<t>"] >= t_end:
                return

            if max_steps is not None and n_steps >= max_steps:
                return

            cur_state = self.next_phase
            try:
                for evt in self.run_single_step():
                    yield evt

            except FailStepException:
                yield StepFailed(t=self.context["<t>"])
                continue

            except TransitionEvent as evt:
                self.next_phase = evt.next_phase

            yield StepCompleted(
                dt=self.context["<dt>"],
                t=self.context["<t>"],
                current_state=cur_state,
                next_phase=self.next_phase)

            n_steps += 1

    def run_single_step(self):
        try:
            self.exec_controller.reset()
            cur_state = self.code.phases[self.next_phase]
            self.next_phase = cur_state.next_phase
            self.exec_controller.update_plan(cur_state, cur_state.depends_on)
            yield from self.exec_controller(cur_state, self)

        finally:
            # discard non-permanent per-step state
            for name in list(self.context.keys()):
                if (
                        not name.startswith("<state>")
                        and not name.startswith("<p>")
                        and name not in ["<t>", "<dt>"]):
                    del self.context[name]

    def register_function(self, name, f):
        if name in self.functions:
            raise ValueError("function '%s' already regsitered" % name)

        self.functions[name] = f

    def evaluate_condition(self, stmt):
        return self.eval_mapper(stmt.condition)

    # {{{ execution methods

    def exec_AssignImplicit(self, stmt):
        raise NotImplementedError("Encountered AssignImplicit.")

    def exec_YieldState(self, stmt):
        return StateComputed(
            t=self.eval_mapper(stmt.time),
            time_id=stmt.time_id,
            component_id=stmt.component_id,
            state_component=self.eval_mapper(stmt.expression)), []

    def exec_Assign(self, stmt):
        if not stmt.loops:
            if stmt.assignee_subscript:
                self.context[stmt.assignee][
                        self.eval_mapper(stmt.assignee_subscript)] = \
                                self.eval_mapper(stmt.expression)
            else:
                self.context[stmt.assignee] = self.eval_mapper(stmt.expression)

        else:
            def implement_loops(loops):
                if not loops:
                    yield
                    return

                ident, start, stop = loops[0]
                for i in range(
                        self.eval_mapper(start), self.eval_mapper(stop)):
                    self.context[ident] = i

                    for _val in implement_loops(loops[1:]):
                        yield

            for _val in implement_loops(stmt.loops):
                if stmt.assignee_subscript:
                    self.context[stmt.assignee][
                            self.eval_mapper(stmt.assignee_subscript)] = \
                                    self.eval_mapper(stmt.expression)
                else:
                    self.context[stmt.assignee] = self.eval_mapper(stmt.expression)

            for ident, _, _ in stmt.loops:
                del self.context[ident]

    def exec_AssignFunctionCall(self, stmt):
        parameters = [
                self.eval_mapper(expr)
                for expr in stmt.parameters]
        kw_parameters = {
                name: self.eval_mapper(expr)
                for name, expr in stmt.kw_parameters.items()}

        func = self.eval_mapper.functions[stmt.function_id]

        results = func(*parameters, **kw_parameters)

        if len(stmt.assignees) == 0:
            return

        if len(stmt.assignees) == 1:
            results = (results,)

        assert len(results) == len(stmt.assignees)

        for assignee, res in zip(stmt.assignees, results):
            self.context[assignee] = res

    def exec_Raise(self, stmt):
        raise stmt.error_condition(stmt.error_message)

    def exec_FailStep(self, stmt):
        raise FailStepException()

    def exec_Nop(self, stmt):
        pass

    def exec_SwitchPhase(self, stmt):
        raise TransitionEvent(stmt.next_phase)

    # }}}

# }}}


# vim: fdm=marker
