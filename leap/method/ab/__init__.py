"""Adams-Bashforth ODE solvers."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
Copyright (C) 2014, 2015 Matt Wala
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
from pymbolic.primitives import CallWithKwargs


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
        from leap.vm.language import TimeIntegratorCode, NewCodeBuilder

        from pymbolic import var

        self.component_id = component_id

        # Declare variables
        self.step = var('<p>step')
        self.function = var('<func>' + component_id)
        self.rhs = var('<p>f_n')
        self.history = \
            [var('<p>f_n_minus_' + str(i)) for i in range(self.order - 1, 0, -1)]
        self.state = var('<state>' + component_id)
        self.t = var('<t>')
        self.dt = var('<dt>')

        # Initialization
        with NewCodeBuilder(label="initialization") as cb:
            cb(self.step, 1)

        cb_init = cb

        steps = self.order

        with NewCodeBuilder(label="primary") as cb:
            cb(self.rhs, self.eval_rhs(self.t, self.state))
            with cb.if_(self.step, "<", steps):
                self.rk_bootstrap(cb)
                cb(self.step, self.step + 1)
            with cb.else_():
                history = self.history + [self.rhs]
                ab_sum = sum(self.coeffs[i] * history[i] for i in range(steps))
                cb(self.state, self.state + self.dt * ab_sum)
                # Rotate history.
                for i in range(len(self.history)):
                    cb.fence()
                    cb(self.history[i], history[i + 1])
            cb(self.t, self.t + self.dt)
            cb.yield_state(expression=self.state, component_id=component_id,
                           time_id='', time=self.t)

        cb_primary = cb

        return TimeIntegratorCode.create_with_init_and_step(
                instructions=cb_init.instructions | cb_primary.instructions,
                initialization_dep_on=cb_init.state_dependencies,
                step_dep_on=cb_primary.state_dependencies,
                step_before_fail=True)

    def eval_rhs(self, t, y):
        """Return a node that evaluates the RHS at the given time and
        component value."""
        return CallWithKwargs(function=self.function,
                              parameters=(),
                              kw_parameters={"t": t, self.component_id: y})

    def rk_bootstrap(self, cb):
        """Initialize the timestepper with an RK method."""

        # Save the current RHS to the AB history

        for i in range(len(self.history)):
            with cb.if_(self.step, "==", i + 1):
                cb(self.history[i], self.rhs)

        rk_tableau, rk_coeffs = self.get_rk_tableau_and_coeffs(self.order)

        # Stage loop (taken from EmbeddedButcherTableauMethod)
        rhss = [var("rk_rhs_" + str(i)) for i in range(len(rk_tableau))]
        for stage_num, (c, coeffs) in enumerate(rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                cb(rhss[stage_num], self.rhs)
            else:
                stage = self.state + sum(self.dt * coeff * rhss[j]
                                         for (j, coeff)
                                         in enumerate(coeffs))

                cb(rhss[stage_num], self.eval_rhs(self.t + c * self.dt, stage))

        # Merge the values of the RHSs.
        rk_comb = sum(coeff * rhss[j] for j, coeff in enumerate(rk_coeffs))
        cb.fence()
        # Assign the value of the new state.
        cb(self.state, self.state + self.dt * rk_comb)
