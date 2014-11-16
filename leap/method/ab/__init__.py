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
from pymbolic import var
from pymbolic.primitives import CallWithKwargs, Comparison
from leap.vm.language import SimpleCodeBuilder

__doc__ = """
.. autoclass:: AdamsBashforthTimeStepper
"""


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


class AdamsBashforthTimeStepper(AdamsBashforthTimeStepperBase):

    def __init__(self, order):
        super(AdamsBashforthTimeStepper, self).__init__()
        self.order = order
        self.coeffs = numpy.asarray(make_ab_coefficients(order))[::-1]

    def __call__(self, component_id):
        from leap.vm.language import If, TimeIntegratorCode, CodeBuilder

        cbuild = CodeBuilder()

        from pymbolic import var

        self.component_id = component_id

        # Declare variables
        self.step = var('<p>step')
        self.rhs = var('<p>f_n')
        self.history = [var('<p>f_n_minus_' + str(i))
                        for i in range(self.order - 1, 0, -1)]
        self.state = var('<state>' + component_id)
        self.t = var('<t>')
        self.dt = var('<dt>')

        # Initialization stage
        with SimpleCodeBuilder(cbuild) as builder:
            builder.assign(self.step, 0)
            builder.assign(self.rhs, self.eval_rhs(self.t, self.state))

        initialization_dep_on = builder.last_added_instruction_id

        # AB stage
        with SimpleCodeBuilder(cbuild) as builder:
            ab_comb = sum(self.coeffs[i] * self.history[i]
                          for i in range(0, self.order - 1))
            ab_comb += self.rhs * self.coeffs[-1]
            builder.assign(self.state, self.state + self.dt * ab_comb)

            # Rotate history
            for i in range(len(self.history)):
                next_val = self.history[i + 1] \
                    if (i + 1) < len(self.history) else self.rhs
                builder.assign(self.history[i], next_val)

            builder.assign(self.rhs,
                           self.eval_rhs(self.t + self.dt, self.state))

        ab_dep_on = builder.last_added_instruction_id

        # Bootstrap stage
        with SimpleCodeBuilder(cbuild) as builder:
            self.rk_bootstrap(builder)
            builder.assign(self.step, self.step + 1)

        bootstrap_dep_on = builder.last_added_instruction_id

        # Main branch
        main_branch = cbuild.add_and_get_ids(
            If(condition=Comparison(self.step, "<", self.order - 1),
               then_depends_on=bootstrap_dep_on,
               else_depends_on=ab_dep_on))

        cbuild.commit()

        # Update and return
        with SimpleCodeBuilder(cbuild, main_branch) as builder:
            builder.yield_state(self.state, component_id,
                                self.t + self.dt, "final")
            builder.assign(self.t, self.t + self.dt)

        step_dep_on = builder.last_added_instruction_id

        return TimeIntegratorCode(instructions=cbuild.instructions,
                                  initialization_dep_on=initialization_dep_on,
                                  step_dep_on=step_dep_on,
                                  step_before_fail=True)

    def eval_rhs(self, t, y):
        """Return a node that evaluates the RHS at the given time and
        component value."""
        return CallWithKwargs(function=var(self.component_id),
                              parameters=(),
                              kw_parameters={"t": t, self.component_id: y})

    def rk_bootstrap(self, builder):
        """Initialize the timestepper with an RK method."""

        # Save the current RHS to the AB history
        for i in range(len(self.history)):
            with builder.condition(Comparison(self.step, "==", i)):
                builder.assign(self.history[i], self.rhs)

        rk_tableau, rk_coeffs = self.get_rk_tableau_and_coeffs(self.order)

        # Stage loop (taken from EmbeddedButcherTableauMethod)
        rhss = [var("rk_rhs_" + str(i)) for i in range(len(rk_tableau))]
        for stage_number, (c, coeffs) in enumerate(rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                builder.assign(rhss[stage_number], self.rhs)
            else:
                stage_state = self.state + sum(self.dt * coeff * rhss[j]
                                               for (j, coeff)
                                               in enumerate(coeffs))

                builder.assign(rhss[stage_number], self.eval_rhs(
                        self.t + c * self.dt, stage_state))

        # Merge the values of the RHSs.
        merged = sum(coeff * rhss[j] for j, coeff in enumerate(rk_coeffs))
        # Assign the value of the new state.
        builder.assign(self.state, self.state + self.dt * merged)
        # Assign the value of the new RHS,
        builder.assign(self.rhs, self.eval_rhs(self.t + self.dt, self.state))
