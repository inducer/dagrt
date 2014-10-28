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

from leap.vm.expression import (DifferentiationMapperWithContext, EvaluationMapper)
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
        self.state = {}
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

        self.eval_mapper = EvaluationMapper(self.state, self.functions)

    def set_up(self, t_start, dt_start, state):
        """
        :arg state: a dictionary mapping state identifiers to their values
        """

        self.state["<t>"] = t_start
        self.state["<dt>"] = dt_start
        for key, val in six.iteritems(state):
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
                    for name in list(six.iterkeys(self.state)):
                        if (
                                not name.startswith("<state>")
                                and not name.startswith("<p>")
                                and name not in ["<t>", "<dt>"]):
                            del self.state[name]

            except FailStepException:
                yield self.StepFailed(t=self.state["<t>"])
                continue

            yield self.StepCompleted(t=self.state["<t>"])

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
        self.state[insn.assignee] = self.eval_mapper(insn.expression)

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


# {{{ step matrix finder

class StepMatrixFinder(NumpyInterpreter):
    """Constructs a step matrix on-the-fly while interpreting code.

    Assumes that all function evaluations occur as the root node of
    a separate assignment instruction.
    """

    def __init__(self, code, function_map, function_deriv_map, variables=None):
        NumpyInterpreter.__init__(self, code, function_map)

        self.function_map = function_map
        self.function_deriv_map = function_deriv_map

        self.diff_states = {}
        if variables is None:
            variables = self.get_state_variables()
        self.variables = variables
        for variable in variables:
            self.diff_states[variable] = {}
        # Initialize the differentiation mapper.
        self.diff_mappers = {}
        for variable in variables:
            context = self.diff_states[variable]
            self.diff_mappers[variable] = \
                DifferentiationMapperWithContext(variable, function_deriv_map,
                context)

    def get_state_variables(self):
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
        from pymbolic import var
        return list(map(var, all_state_vars))

    def build_step_matrix(self):
        nv = len(self.variables)
        step_matrix = np.zeros((nv, nv),)
        for i, v in enumerate(self.variables):
            for j, vv in enumerate(self.variables):
                step_matrix[i][j] = self.diff_mappers[vv](v)
        return step_matrix

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
                        if isinstance(event, self.StateComputed):
                            event.step_matrix = self.build_step_matrix()
                            # Discard computed derivatives.
                            for variable in self.variables:
                                self.diff_states[variable].clear()
                        yield event
                finally:
                    # discard non-permanent per-step state
                    for name in list(six.iterkeys(self.state)):
                        if (
                                not name.startswith("<state>")
                                and not name.startswith("<p>")
                                and name not in ["<t>", "<dt>"]):
                            del self.state[name]

            except FailStepException:
                yield self.StepFailed(t=self.state["<t>"])
                continue

            yield self.StepCompleted(t=self.state["<t>"])

            if last_step:
                break

    def _exec_func_eval_assignment(self, assignee, call):
        func = self.function_map[call.function.name]

        from pymbolic.primitives import CallWithKwargs
        evaluated_args = [
            self.eval_mapper(arg_expr)
            for name, arg_expr in call.parameters]

        if isinstance(call, CallWithKwargs):
            evaluated_kwargs = [
                    (name, self.eval_mapper(arg_expr))
                    for name, arg_expr in call.kw_parameters.items()]
        else:
            evaluated_kwargs = []

        self.state[assignee] = func(*evaluated_args, **dict(evaluated_kwargs))

        # Compute derivatives of assignee by chain rule.
        func_deriv = self.function_deriv_map[call.function.name]

        for variable in self.variables:
            total_deriv = 0

            for i, arg in enumerate(call.parameters):
                deriv = self.diff_mappers[variable](arg)
                eval_deriv = self.eval_mapper(deriv)
                total_deriv += (
                        func_deriv(i, *evaluated_args, **dict(evaluated_kwargs))
                        * eval_deriv)

            if isinstance(call, CallWithKwargs):
                for arg_name, arg in call.kw_parameters.items():
                    deriv = self.diff_mappers[variable](arg)
                    eval_deriv = self.eval_mapper(deriv)
                    total_deriv += (
                            func_deriv(arg_name,
                                *evaluated_args, **dict(evaluated_kwargs))
                            * eval_deriv)

            self.diff_states[variable][assignee] = self.eval_mapper(total_deriv)

    def exec_AssignExpression(self, insn):
        from pymbolic.primitives import Call, CallWithKwargs

        if isinstance(insn.expression, (Call, CallWithKwargs)):
            self._exec_func_eval_assignment(insn.assignee, insn.expression)
        else:
            self.state[insn.assignee] = self.eval_mapper(insn.expression)
            for variable in self.variables:
                deriv = self.diff_mappers[variable](insn.expression)
                self.diff_states[variable][insn.assignee] = self.eval_mapper(deriv)

# }}}

# vim: fdm=marker
