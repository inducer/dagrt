"""Runge-Kutta ODE timestepper."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007-2013 Andreas Kloeckner
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

import numpy as np
from leap.method import Method, TimeStepUnderflow
from leap.vm.language import CodeBuilder


__doc__ = """
.. autoclass:: ODE23TimeStepper

.. autoclass:: ODE45TimeStepper

.. autoclass:: LSRK4TimeStepper
"""


def verify_first_same_as_last_condition(times, last_stage_coefficients,
                                        output_stage_coefficients):
    if not times or not last_stage_coefficients \
            or not output_stage_coefficients:
        return False
    if times[0] != 0:
        return False

    def truncate_final_zeros(array):
        index = len(array) - 1
        while array[index] == 0 and index >= 0:
            index -= 1
        return array[:index + 1]

    if truncate_final_zeros(last_stage_coefficients) != \
           truncate_final_zeros(output_stage_coefficients):
        return False
    return True


# {{{ Embedded Runge-Kutta schemes base class

class EmbeddedRungeKuttaMethod(Method):

    def __init__(self, use_high_order=True,
            atol=0, rtol=0, max_dt_growth=5, min_dt_shrinkage=0.1,
            limiter_name=None):

        self.limiter_name = limiter_name

        self.use_high_order = use_high_order

        self.adaptive = bool(atol or rtol)
        self.atol = atol
        self.rtol = rtol

        self.max_dt_growth = max_dt_growth
        self.min_dt_shrinkage = min_dt_shrinkage

    def finish_adaptive(self, cb, high_order_estimate, high_order_rhs,
                        low_order_estimate):
        from pymbolic import var
        from pymbolic.primitives import Comparison, LogicalOr, Max, Min
        from leap.vm.expression import IfThenElse

        norm_start_state = var('norm_start_state')
        norm_end_state = var('norm_end_state')
        rel_error_raw = var('rel_error_raw')
        rel_error = var('rel_error')

        cb.fence()

        norm = lambda expr: var('<builtin>norm_2')(expr)
        cb(norm_start_state, norm(self.state))
        cb(norm_end_state, norm(low_order_estimate))
        cb(rel_error_raw, norm((high_order_estimate - low_order_estimate) /
                               (var('<builtin>len')(self.state) ** 0.5 *
                                (self.atol + self.rtol *
                                 Max((norm_start_state, norm_end_state))
                                 ))))

        cb(rel_error, IfThenElse(Comparison(rel_error_raw, "==", 0),
                                 1.0e-14, rel_error_raw))

        with cb.if_(LogicalOr((Comparison(rel_error, ">", 1),
                               var('<builtin>isnan')(rel_error)))):

            with cb.if_(var('<builtin>isnan')(rel_error)):
                cb(self.dt, self.min_dt_shrinkage * self.dt)
            with cb.else_():
                cb(self.dt, Max((0.9 * self.dt *
                                 rel_error ** (-1 / self.low_order),
                                 self.min_dt_shrinkage * self.dt)))

            with cb.if_(self.t + self.dt, '==', self.t):
                cb.raise_(TimeStepUnderflow)
            with cb.else_():
                cb.fail_step()

        with cb.else_():
            if self.limiter is not None:
                cb(high_order_estimate, self.limiter(high_order_estimate))

            self.finish_nonadaptive(cb, high_order_estimate, high_order_rhs,
                                    low_order_estimate)
            cb.fence()
            cb(self.dt,
               Min((0.9 * self.dt * rel_error ** (-1 / self.high_order),
                    self.max_dt_growth * self.dt)))


class EmbeddedButcherTableauMethod(EmbeddedRungeKuttaMethod):
    """
    User-supplied context:
        <state> + component_id: The value that is integrated
        <func> + component_id: The right hand side function
    """

    def __init__(self, component_id, *args, **kwargs):
        """
        :arg component_id: an identifier to be used for the single state
            component supported.
        """
        EmbeddedRungeKuttaMethod.__init__(self, *args, **kwargs)

        # Set up variables.
        from pymbolic import var

        self.component_id = component_id

        self.dt = var("<dt>")
        self.t = var("<t>")
        self.last_rhs = var("<p>last_rhs_" + component_id)
        self.state = var("<state>" + component_id)
        self.rhs_func = var("<func>" + component_id)

        if self.limiter_name is not None:
            self.limiter = var("<func>" + self.limiter_name)
        else:
            self.limiter = None

    def generate(self):
        from leap.vm.language import TimeIntegratorCode
        from pymbolic import var

        # Initialization.

        with CodeBuilder("initialization") as cb:
            cb(self.last_rhs, self.call_rhs(self.t, self.state))

        cb_init = cb

        # Primary.

        with CodeBuilder("primary") as cb:
            local_last_rhs = var('last_rhs_' + self.component_id)
            rhss = []
            # {{{ stage loop
            last_stage = len(self.butcher_tableau) - 1
            for stage_num, (c, coeffs) in enumerate(self.butcher_tableau):
                if len(coeffs) == 0:
                    assert c == 0
                    cb(local_last_rhs, self.last_rhs)
                    this_rhs = local_last_rhs
                else:
                    stage_state = (
                        self.state + sum(
                            self.dt * coeff * rhss[j]
                            for j, coeff in enumerate(coeffs)))

                    if self.limiter is not None:
                        stage_state = self.limiter({self.component_id: stage_state})

                    if stage_num == last_stage:
                        high_order_estimate = var("high_order_estimate")
                        cb(high_order_estimate, stage_state)

                    this_rhs = var("rhs_" + str(stage_num))
                    cb(this_rhs, self.call_rhs(self.t + c * self.dt,
                                               high_order_estimate
                                                   if stage_num == last_stage
                                                   else stage_state))
                rhss.append(this_rhs)
            # }}}

            low_order_estimate = var('low_order_estimate')
            cb(low_order_estimate, self.state +
               sum(self.dt * coeff * rhss[j] for j, coeff in
                   enumerate(self.low_order_coeffs)))

            if not self.adaptive:
                self.finish_nonadaptive(cb, high_order_estimate, rhss[-1],
                                        low_order_estimate)
            else:
                self.finish_adaptive(cb, high_order_estimate, rhss[-1],
                                     low_order_estimate)

        cb_primary = cb

        return TimeIntegratorCode.create_with_init_and_step(
            instructions=cb_init.instructions | cb_primary.instructions,
            initialization_dep_on=cb_init.state_dependencies,
            step_dep_on=cb_primary.state_dependencies)

    def finish_nonadaptive(self, cb, high_order_estimate, high_order_rhs,
                           low_order_estimate):
        cb.fence()
        if not self.use_high_order:
            cb(self.last_rhs,
               self.call_rhs(self.t + self.dt, low_order_estimate))
            cb(self.state, low_order_estimate)
        else:
            times, coeffs = tuple(zip(*self.butcher_tableau))
            assert verify_first_same_as_last_condition(
                    times, coeffs[-1], self.high_order_coeffs)
            cb(self.last_rhs, high_order_rhs)
            cb(self.state, high_order_estimate)

        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')
        cb.fence()
        cb(self.t, self.t + self.dt)

    def call_rhs(self, t, y):
        return self.rhs_func(t=t, **{self.component_id: y})

# }}}


# {{{ Bogacki-Shampine second/third-order Runge-Kutta

class ODE23TimeStepper(EmbeddedButcherTableauMethod):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    http://dx.doi.org/10.1016/0893-9659(89)90079-7
    """

    butcher_tableau = [
            (0, []),
            (1/2, [1/2]),
            (3/4, [0, 3/4]),
            (1, [2/9, 1/3, 4/9])
            ]

    low_order = 2
    low_order_coeffs = [7/24, 1/4, 1/3, 1/8]
    high_order = 3
    high_order_coeffs = [2/9, 1/3, 4/9, 0]

# }}}


# {{{ Dormand-Prince fourth/fifth-order Runge-Kutta

class ODE45TimeStepper(EmbeddedButcherTableauMethod):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    http://dx.doi.org/10.1016/0771-050X(80)90013-3.
    """

    butcher_tableau = [
            (0, []),
            (1/5, [1/5]),
            (3/10, [3/40, 9/40]),
            (4/5, [44/45, -56/15, 32/9]),
            (8/9, [19372/6561, -25360/2187, 64448/6561, -212/729]),
            (1, [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
            (1, [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
            ]

    low_order = 4
    low_order_coeffs = [5179/57600, 0, 7571/16695, 393/640, -92097/339200,
            187/2100, 1/40]
    high_order = 5
    high_order_coeffs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# }}}


# {{{ Carpenter/Kennedy low-storage fourth-order Runge-Kutta

class LSRK4TimeStepper(Method):
    """A low storage fourth-order Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.64
    or
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994
    """

    _RK4A = [
            0.0,
            -567301805773 / 1357537059087,
            -2404267990393 / 2016746695238,
            -3550918686646 / 2091501179385,
            -1275806237668 / 842570457699,
            ]

    _RK4B = [
            1432997174477 / 9575080441755,
            5161836677717 / 13612068292357,
            1720146321549 / 2090206949498,
            3134564353537 / 4481467310338,
            2277821191437 / 14882151754819,
            ]

    _RK4C = [
            0.0,
            1432997174477/9575080441755,
            2526269341429/6820363962896,
            2006345519317/3224310063776,
            2802321613138/2924317926251,
            #1,
            ]
    coeffs = np.array([_RK4A, _RK4B, _RK4C]).T

    adaptive = False

    def __init__(self, component_id, limiter_name=None):
        """
        :arg component_id: an identifier to be used for the single state
            component supported.
        """

        # Set up variables.
        from pymbolic import var

        self.component_id = component_id

        if limiter_name is not None:
            self.limiter = var("<func>" + self.limiter_name)
        else:
            self.limiter = None

    def generate(self):
        comp_id = self.component_id

        from pymbolic import var
        dt = var("<dt>")
        t = var("<t>")
        residual = var("<p>residual_" + comp_id)
        state = var("<state>" + comp_id)
        rhs_func = var("<func>" + comp_id)

        with CodeBuilder("initialization") as cb:
            cb(residual, 0)

        cb_init = cb

        # Primary.

        rhs_val = var("rhs_val")

        with CodeBuilder("primary") as cb:
            for a, b, c in self.coeffs:
                cb.fence()
                cb(rhs_val, rhs_func(t=t + c*dt, **{comp_id: state}))
                cb(residual, a*residual + dt*rhs_val)
                new_state_expr = state + b * residual

                if self.limiter is not None:
                    new_state_expr = self.limiter({comp_id: new_state_expr})

                cb.fence()
                cb(state, new_state_expr)

            cb.yield_state(state, comp_id, t + dt, 'final')
            cb.fence()
            cb(t, t + dt)

        cb_primary = cb

        from leap.vm.language import TimeIntegratorCode
        return TimeIntegratorCode.create_with_init_and_step(
            instructions=cb_init.instructions | cb_primary.instructions,
            initialization_dep_on=cb_init.state_dependencies,
            step_dep_on=cb_primary.state_dependencies)

# }}}


# vim: foldmethod=marker
