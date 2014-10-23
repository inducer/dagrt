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
import numpy.linalg as la
import scipy.optimize

from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.differentiator import DifferentiationMapper as \
    DifferentiationMapperBase

import six


class FailStepException(Exception):
    pass


class EvaluationMapper(EvaluationMapperBase):

    def __init__(self, context, functions):
        """
        :arg context: a mapping from variable names to values
        :arg functions: a mapping from function names to functions
        """
        EvaluationMapperBase.__init__(self, context)
        self.functions = functions

    def handle_call(self, function_name, parameters, kw_parameters):
        if function_name in self.functions:
            function = self.functions[function_name]
        else:
            raise ValueError("Call to unknown function: " + str(function_name))
        evaluated_parameters = (self.rec(param) for param in parameters)
        evaluated_kw_parameters = {param_id: self.rec(param)
             for param_id, param in six.iteritems(kw_parameters)}
        return function(*evaluated_parameters, **evaluated_kw_parameters)

    def map_call(self, expr):
        return self.handle_call(expr.function.name, expr.parameters, {})

    def map_call_with_kwargs(self, expr):
        return self.handle_call(expr.function.name, expr.parameters,
                                expr.kw_parameters)


class DifferentiationMapperWithContext(DifferentiationMapperBase):

    def __init__(self, variable, functions, context):
        DifferentiationMapperBase.__init__(self, variable, None)
        self.context = context
        self.functions = functions

    def map_call(self, expr):
        raise NotImplementedError

    def map_variable(self, expr):
        return self.context[expr.name] if expr.name in self.context else \
            DifferentiationMapperBase.map_variable(self, expr)


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
        builtins = {"len": len, "isnan": np.isnan}

        assert not set(builtins.keys()) & set(rhs_map.keys())

        self.functions = dict(builtins, **rhs_map)
        self.rhs_map = rhs_map

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

    def exec_AssignRHS(self, insn):
        rhs = self.rhs_map[insn.component_id]
        t = self.eval_mapper(insn.t)

        for assignee, args in zip(insn.assignees, insn.rhs_arguments):
            self.state[assignee] = rhs(t, **dict(
                    (name, self.eval_mapper(expr))
                    for name, expr in args))

    def exec_AssignSolvedRHS(self, insn):

        class FunctionWithContext(object):

            def __init__(self, expression, arg_name, context, functions):
                self.eval_mapper = EvaluationMapper(self, functions)
                self.expression = expression
                self.arg_name = arg_name
                self.context = context

            def __call__(self, arg):
                self.value = arg
                return self.eval_mapper(self.expression)

            def __getitem__(self, name):
                if name == self.arg_name:
                    return self.value
                else:
                    return self.context[name]

        func = FunctionWithContext(insn.expressions[0],
                                   insn.solve_components[0].name, self.state,
                                   self.functions)

        if insn.solver_id == 'newton':
            guess = self.eval_mapper(insn.solver_parameters['initial_guess'])
            self.state[insn.assignees[0]] = scipy.optimize.newton(func, guess)
        else:
            raise ValueError('Unknown solver id: ' + str(insn.solver_id))

    def exec_ReturnState(self, insn):
        return self.StateComputed(
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


class StepMatrixFinder(NumpyInterpreter):
    """Constructs a step matrix on-the-fly while interpreting code."""

    def __init__(self, code, rhs_map, rhs_deriv_map, variables=None):
        NumpyInterpreter.__init__(self, code, rhs_map)
        self.rhs_deriv_map = rhs_deriv_map
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
                DifferentiationMapperWithContext(variable, rhs_deriv_map,
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

    def exec_AssignRHS(self, insn):
        rhs = self.rhs_map[insn.component_id]
        t = self.eval_mapper(insn.t)

        evaluated = []

        for assignee, args in zip(insn.assignees, insn.rhs_arguments):
            rhsargs = [(name, self.eval_mapper(expr)) for name, expr in args]
            evaluated.append(rhsargs)
            self.state[assignee] = rhs(t, **dict(rhsargs))

        # Compute derivatives of assignee by chain rule.
        rhs_deriv = self.rhs_deriv_map[insn.component_id]

        for assignee, args, ev_args in \
                zip(insn.assignees, insn.rhs_arguments, evaluated):
            for variable in self.variables:
                total_deriv = 0
                for n, arg in enumerate(args):
                    deriv = self.diff_mappers[variable](arg[1])
                    eval_deriv = self.eval_mapper(deriv)
                    total_deriv += rhs_deriv(1 + n, t, **dict(ev_args)) * eval_deriv
                self.diff_states[variable][assignee] = self.eval_mapper(total_deriv)

    def exec_AssignExpression(self, insn):
        self.state[insn.assignee] = self.eval_mapper(insn.expression)
        for variable in self.variables:
            deriv = self.diff_mappers[variable](insn.expression)
            self.diff_states[variable][insn.assignee] = self.eval_mapper(deriv)

# vim: fdm=marker
