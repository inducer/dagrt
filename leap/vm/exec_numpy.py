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

from collections import namedtuple
import numpy as np
from leap.vm.expression import EvaluationMapper
import six


class FailStepException(Exception):
    pass


class TransitionEvent(Exception):

    def __init__(self, next_state):
        self.next_state = next_state


# {{{ interpreter

class NumpyInterpreter(object):
    """A :mod:`numpy`-targeting interpreter for the time integration language
    defined in :mod:`leap.vm.language`.

    .. attribute:: next_state

    .. automethod:: set_up
    .. automethod:: run
    .. automethod:: run_single_step
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

    class StepCompleted(
            namedtuple("StepCompleted",
                ["t", "current_state", "next_state"])):
        """
        .. attribute:: t

            Approximate integrator time at end of step.

        .. attribute:: current_state
        .. attribute:: next_state
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

        def builtin_norm(x, ord):
            if np.isscalar(x):
                return abs(x)
            return np.linalg.norm(x, ord)

        def builtin_array(n):
            if n != np.floor(n):
                raise ValueError("array() argument n is not an integer")
            n = int(n)

            return np.empty(n, dtype=np.float64)

        def builtin_matmul(a, b, a_cols, b_cols):
            if a_cols != np.floor(a_cols):
                raise ValueError("matmul() argument a_cols is not an integer")
            if b_cols != np.floor(b_cols):
                raise ValueError("matmul() argument b_cols is not an integer")
            a_cols = int(a_cols)
            b_cols = int(b_cols)

            a_mat = a.reshape(-1, a_cols, order="F")
            b_mat = b.reshape(-1, b_cols, order="F")

            res_mat = a_mat.dot(b_mat)

            return res_mat.reshape(-1, order="F")

        def builtin_linear_solve(a, b, a_cols, b_cols):
            if a_cols != np.floor(a_cols):
                raise ValueError("linear_solve() argument a_cols is not an integer")
            if b_cols != np.floor(b_cols):
                raise ValueError("linear_solve() argument b_cols is not an integer")
            a_cols = int(a_cols)
            b_cols = int(b_cols)

            a_mat = a.reshape(-1, a_cols, order="F")
            b_mat = b.reshape(-1, b_cols, order="F")

            import numpy.linalg as la
            res_mat = la.solve(a_mat, b_mat)

            return res_mat.reshape(-1, order="F")

        from functools import partial

        builtins = {
                "<builtin>len": np.size,
                "<builtin>isnan": np.isnan,
                "<builtin>norm_1": partial(builtin_norm, ord=1),
                "<builtin>norm_2": partial(builtin_norm, ord=2),
                "<builtin>norm_inf": partial(builtin_norm, ord=np.inf),
                "<builtin>dot_product": np.vdot,
                "<builtin>array": builtin_array,
                "<builtin>matmul": builtin_matmul,
                "<builtin>linear_solve": builtin_linear_solve,
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

    def run(self, t_end=None, max_steps=None):
        """Generates :ref:`numpy-exec-events`."""

        n_steps = 0
        while True:
            if t_end is not None and self.context["<t>"] >= t_end:
                return

            if max_steps is not None and n_steps >= max_steps:
                return

            cur_state = self.next_state
            try:
                for evt in self.run_single_step():
                    yield evt

            except FailStepException:
                yield self.StepFailed(t=self.context["<t>"])
                continue

            except TransitionEvent as evt:
                self.next_state = evt.next_state

            yield self.StepCompleted(
                    t=self.context["<t>"],
                    current_state=cur_state,
                    next_state=self.next_state)

            n_steps += 1

    def run_single_step(self):
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

    def register_function(self, name, f):
        if name in self.functions:
            raise ValueError("function '%s' already regsitered" % name)

        self.functions[name] = f

    def evaluate_condition(self, insn):
        return self.eval_mapper(insn.condition)

    # {{{ execution methods

    def exec_AssignSolved(self, insn):
        raise NotImplementedError("Encountered AssignSolved.")

    def exec_YieldState(self, insn):
        return self.StateComputed(
                    t=self.eval_mapper(insn.time),
                    time_id=insn.time_id,
                    component_id=insn.component_id,
                    state_component=self.eval_mapper(insn.expression)), []

    def exec_AssignExpression(self, insn):
        if not insn.loops:
            if insn.assignee_subscript:
                self.context[insn.assignee][
                        self.eval_mapper(insn.assignee_subscript)] = \
                                self.eval_mapper(insn.expression)
            else:
                self.context[insn.assignee] = self.eval_mapper(insn.expression)

        else:
            def implement_loops(loops):
                if not loops:
                    yield
                    return

                ident, start, stop = loops[0]
                for i in six.moves.range(
                        self.eval_mapper(start), self.eval_mapper(stop)):
                    self.context[ident] = i

                    for val in implement_loops(loops[1:]):
                        yield

            for val in implement_loops(insn.loops):
                if insn.assignee_subscript:
                    self.context[insn.assignee][
                            self.eval_mapper(insn.assignee_subscript)] = \
                                    self.eval_mapper(insn.expression)
                else:
                    self.context[insn.assignee] = self.eval_mapper(insn.expression)

            for ident, _, _ in insn.loops:
                del self.context[ident]

    def exec_Raise(self, insn):
        raise insn.error_condition(insn.error_message)

    def exec_FailStep(self, insn):
        raise FailStepException()

    def exec_Nop(self, insn):
        pass

    def exec_StateTransition(self, insn):
        raise TransitionEvent(insn.next_state)

    # }}}

# }}}


# {{{ step matrix finder

class StepMatrixFinder(object):
    """Constructs a step matrix on-the-fly while interpreting code.

    Assumes that all function evaluations occur as the root node of
    a separate assignment instruction.
    """

    def __init__(self, code, function_map, variables=None):
        self.code = code

        self.function_map = function_map

        if variables is None:
            variables = self._get_state_variables()
        self.variables = variables

        from leap.vm.language import ExecutionController
        self.exec_controller = ExecutionController(code)
        self.context = {}

        self.eval_mapper = EvaluationMapper(self.context, self.function_map)

    def _get_state_variables(self):
        """Extract all state-related variables from the code."""
        all_var_ids = set()
        for inst in self.code.instructions:
            all_var_ids |= inst.get_assignees()
            all_var_ids |= inst.get_read_variables()
        all_state_vars = []
        for var_name in all_var_ids:
            if var_name.startswith('<p>') or var_name.startswith('<state>'):
                all_state_vars.append(var_name)
        all_state_vars.sort()
        return all_state_vars

    def get_state_step_matrix(self, state_name):
        state = self.code.states[state_name]

        from pymbolic import var

        initial_vars = []

        self.context.clear()
        for vname in self.variables:
            iv = self.context[vname] = var(vname+"_0")
            initial_vars.append(iv)

        self.context["<dt>"] = var("<dt>")
        self.context["<t>"] = 0

        self.exec_controller.reset()
        self.exec_controller.update_plan(state.depends_on)
        for event in self.exec_controller(self):
            pass

        from pymbolic.mapper.differentiator import DifferentiationMapper

        nv = len(self.variables)
        step_matrix = np.zeros((nv, nv), dtype=np.object)
        for i, v in enumerate(self.variables):
            for j, iv in enumerate(initial_vars):
                step_matrix[i][j] = DifferentiationMapper(iv)(self.context[v])
        return step_matrix

    def evaluate_condition(self, insn):
        if insn.condition is not True:
            raise RuntimeError("matrices don't represent conditionals well, "
                "so StepMatrixFinder cannot support them")
        return True

    # {{{ exec methods

    def exec_AssignExpression(self, insn):
        self.context[insn.assignee] = self.eval_mapper(insn.expression)

    def exec_Nop(self, insn):
        pass

    def exec_YieldState(self, insn):
        pass

    def exec_Raise(self, insn):
        raise insn.error_condition(insn.error_message)

    def exec_FailStep(self, insn):
        raise FailStepException()

    def exec_StateTransition(self, insn):
        pass

    # }}}

# }}}

# vim: fdm=marker
