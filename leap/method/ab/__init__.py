
"""Adams-Bashforth ODE solvers."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
Copyright (C) 2014 Matt Wala
"""

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

import numpy
from leap.method.ab.utils import make_ab_coefficients
from leap.method import Method
from pymbolic.primitives import CallWithKwargs


class AdamsBashforthTimeStepperBase(Method):

    @staticmethod
    def get_rk_tableau_and_coeffs(order):
        from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper
        # TODO: Move the tabular data to its own module.
        if order <= 2:
            rk_tableau = ODE23TimeStepper.butcher_tableau
            rk_coeffs = ODE23TimeStepper.low_order_coeffs
        elif order == 3:
            rk_tableau = ODE23TimeStepper.butcher_tableau
            rk_coeffs = ODE23TimeStepper.high_order_coeffs
        elif order == 4:
            rk_tableau = ODE45TimeStepper.butcher_tableau
            rk_coeffs = ODE45TimeStepper.low_order_coeffs
        elif order == 5:
            rk_tableau = ODE45TimeStepper.butcher_tableau
            rk_coeffs = ODE45TimeStepper.high_order_coeffs
        else:
            raise ValueError('Unsupported order: %s' % order)
        return (rk_tableau, rk_coeffs)

    def emit_epilogue(self, cbuild, return_val, component_id, depends_on=[]):
        """Add code that runs at the end of the timestep."""

        from leap.vm.language import ReturnState, AssignExpression
        from pymbolic import var

        return cbuild.add_and_get_ids(ReturnState(
            id='ret', time_id='final',
            time=var('<t>') + var('<dt>'),
            component_id=component_id,
            expression=return_val,
            depends_on=depends_on,
            ),
            AssignExpression('<t>', var('<t>') + var('<dt>'), id='increment_t',
                             depends_on=depends_on + ['ret']))

    def emit_rk_body(self, cbuild, component_id, order, state, rhs,
                     t, dt, depends_on=[], label=''):
        """Emits the code for a Runge-Kutta method

        Inputs
            cbuild:     the CodeBuilder
            component_id:   the function to call
            order:      the order of accuracy for the method
            state:      the state to use / update
            rhs:        the RHS to use / update
            t:          the time variable
            dt:         the timestep value
            depends_on: any instructions to execute before
            label:      the label to prefix names with

        Returns a pair [y, rhs] of instruction names that compute the new value
        of y and the rhs at t + dt."""

        from leap.vm.language import AssignExpression
        from pymbolic import var

        add_and_get_ids = cbuild.add_and_get_ids

        rk_tableau, rk_coeffs = self.get_rk_tableau_and_coeffs(order)

        # Stage loop (taken from EmbeddedButcherTableauMethod)

        rhss = []

        all_rhs_eval_ids = []

        for (istage, (c, coeffs)) in enumerate(rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                this_rhs = rhs
            else:
                stage_state = state + sum(dt * coeff * rhss[j] for (j,
                        coeff) in enumerate(coeffs))

                rhs_id = cbuild.fresh_var_name(label + ('_rhs%d' % istage) + "_")
                rhs_insn_id = cbuild.fresh_insn_id(label+('_ev_rhs%d' % istage)+"_")

                add_and_get_ids(AssignExpression(
                        assignee=var(rhs_id),
                        expression=var(component_id)(t=t + c * dt,
                                                     y=stage_state),
                        depends_on=depends_on + all_rhs_eval_ids,
                        id=rhs_insn_id))

                all_rhs_eval_ids.append(rhs_insn_id)
                this_rhs = var(rhs_id)
            rhss.append(this_rhs)

        update_state_id = cbuild.fresh_insn_id(label + '_update_state')

        # Merge the values of the RHSs

        merged_rhss = var(label + '_merged_rhss')

        merge_id, = add_and_get_ids(AssignExpression(merged_rhss.name,
                        (sum(coeff * rhss[j] for (j, coeff) in
                        enumerate(rk_coeffs))), depends_on=all_rhs_eval_ids))

        # Assign the value of the new state

        add_and_get_ids(AssignExpression(state.name, state + dt * merged_rhss,
                        id=update_state_id, depends_on=[merge_id]))

        # Assign the value of the new RHS

        update_rhs_id, = \
            add_and_get_ids(AssignExpression(rhs.name, this_rhs,
                            depends_on=[update_state_id]))

        return [update_state_id, update_rhs_id]


class AdamsBashforthTimeStepper(AdamsBashforthTimeStepperBase):

    def __init__(self, order):
        super(AdamsBashforthTimeStepper, self).__init__()
        self.order = order
        self.coeffs = numpy.asarray(make_ab_coefficients(order))[::-1]

    def rk_bootstrap(self, cbuild, component_id):
        """Initialize the timestepper with an RK method."""

        from leap.vm.language import AssignExpression, If

        add_and_get_ids = cbuild.add_and_get_ids
        from pymbolic import var

        step = var('<p>step')
        fvals = [var('<p>last_rhs_%d' % i) for i in
                 range(self.order - 1, 0, -1)]
        state = var('<state>' + component_id)
        t = var('<t>')
        dt = var('<dt>')
        last_rhs = var('<p>last_rhs_' + component_id)

        # Save the current RHS to the AB history

        condition_ids = []

        for (i, fval) in enumerate(fvals):
            from pymbolic.primitives import Comparison
            assign_id, = add_and_get_ids(AssignExpression(fval.name, last_rhs))
            condition_id, = \
                add_and_get_ids(If(condition=Comparison(step, '==', i),
                                then_depends_on=[assign_id],
                                else_depends_on=[]))
            condition_ids.append(condition_id)

        # Compute the new value of the state and RHS

        rk = self.emit_rk_body(cbuild, component_id, self.order,
                               state, last_rhs, t, dt, depends_on=condition_ids)

        return rk

    def __call__(self, component_id):
        from leap.vm.language import AssignExpression, If, \
            TimeIntegratorCode, CodeBuilder

        cbuild = CodeBuilder()
        add_and_get_ids = cbuild.add_and_get_ids

        from pymbolic import var

        # Declare variables

        step = var('<p>step')
        fvals = [var('<p>last_rhs_%d' % i) for i in
                 range(self.order - 1, 0, -1)]
        state = var('<state>' + component_id)
        t = var('<t>')
        dt = var('<dt>')
        last_rhs = var('<p>last_rhs_' + component_id)
        curr_rhs = var('curr_rhs')

        dep_inf_exclude_names = ['<t>', '<dt>', state.name, last_rhs.name,
            step.name]

        # Initialize variables

        initialization_dep_on = \
            add_and_get_ids(AssignExpression(step.name, 0),
                            AssignExpression(assignee=last_rhs.name,
                                expression=CallWithKwargs(
                                    function=var(component_id),
                                    parameters=(), kw_parameters={'t': t,
                                    component_id: state})))
        cbuild.commit()

        # RK bootstrap stage

        bootstrap_ids = self.rk_bootstrap(cbuild, component_id)

        add_and_get_ids(AssignExpression(step.name, step + 1,
                        id='increment_step'))

        cbuild.infer_single_writer_dependencies(exclude=dep_inf_exclude_names)
        cbuild.commit()

        # AB stage

        add_and_get_ids(
            AssignExpression(assignee=curr_rhs.name,
                expression=CallWithKwargs(function=var(component_id),
                    parameters=(),
                    kw_parameters={'t': t, component_id: state}),
                id='compute_curr_rhs'),
            AssignExpression(state.name, state + dt *
                (sum(self.coeffs[i] * fvals[i] for i in range(0, self.order-1))
                 + curr_rhs * self.coeffs[-1]), id='ab_update_state',
                depends_on=['compute_curr_rhs'])
            )

        cbuild.commit()

        # Update AB history

        last_dep_id = 'ab_update_state'

        for i, fval in enumerate(fvals):
            next_fval = fvals[i + 1] if i + 1 < len(fvals) else curr_rhs
            last_dep_id, = add_and_get_ids(AssignExpression(fval.name,
                            next_fval, depends_on=[last_dep_id]))

        # The branch to decide whether the current step is an initialization
        # step or an AB timestepping step

        from pymbolic.primitives import Comparison

        main_branch_id, = \
            add_and_get_ids(If(condition=Comparison(step, '<',
                            self.order - 1), then_depends_on=bootstrap_ids +
                            ['increment_step'], else_depends_on=[last_dep_id]))

        # Increment t and return the state

        epilogue = self.emit_epilogue(cbuild, state, component_id,
                                      [main_branch_id])

        cbuild.commit()

        return TimeIntegratorCode(instructions=cbuild.instructions,
                                  initialization_dep_on=initialization_dep_on,
                                  step_dep_on=epilogue,
                                  step_before_fail=False)
