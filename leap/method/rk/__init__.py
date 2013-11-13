"""Runge-Kutta ODE timestepper."""

from __future__ import division

__copyright__ = "Copyright (C) 2007-2013 Andreas Kloeckner"

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
from leap.method import Method


# {{{ Embedded Runge-Kutta schemes base class

def adapt_step_size(t, dt,
        start_y, high_order_end_y, low_order_end_y, stepper, lc2, norm):
    normalization = stepper.atol + stepper.rtol*max(
                norm(low_order_end_y), norm(start_y))

    error = lc2(
        (1/normalization, high_order_end_y),
        (-1/normalization, low_order_end_y)
        )

    if rel_err == 0:
        rel_err = 1e-14

    if rel_err > 1 or numpy.isnan(rel_err):
        # reject step

        if not numpy.isnan(rel_err):
            dt = max(
                    0.9 * dt * rel_err**(-1/stepper.low_order),
                    stepper.min_dt_shrinkage * dt)
        else:
            dt = stepper.min_dt_shrinkage*dt

        if t + dt == t:
            from hedge.timestep import TimeStepUnderflow
            raise TimeStepUnderflow()

        return False, dt, rel_err
    else:
        # accept step

        next_dt = min(
                0.9 * dt * rel_err**(-1/stepper.high_order),
                stepper.max_dt_growth*dt)

        return True, next_dt, rel_err


class EmbeddedRungeKuttaMethod(Method):
    def __init__(self, use_high_order=True, dtype=numpy.float64, rcon=None,
            vector_primitive_factory=None, atol=0, rtol=0,
            max_dt_growth=5, min_dt_shrinkage=0.1,
            limiter=None):
        if vector_primitive_factory is None:
            from hedge.vector_primitives import VectorPrimitiveFactory
            self.vector_primitive_factory = VectorPrimitiveFactory()
        else:
            self.vector_primitive_factory = vector_primitive_factory

        from pytools.log import IntervalTimer, EventCounter
        timer_factory = IntervalTimer
        if rcon is not None:
            timer_factory = rcon.make_timer

        if limiter is None:
            self.limiter = lambda x: x
        else:
            self.limiter = limiter

        self.timer = timer_factory(
                "t_rk", "Time spent doing algebra in Runge-Kutta")
        self.flop_counter = EventCounter(
                "n_flops_rk", "Floating point operations performed in Runge-Kutta")

        self.use_high_order = use_high_order

        self.dtype = numpy.dtype(dtype)

        self.adaptive = bool(atol or rtol)
        self.atol = atol
        self.rtol = rtol

        from pytools import match_precision
        self.scalar_dtype = match_precision(
                numpy.dtype(numpy.float64), self.dtype)

        self.max_dt_growth = max_dt_growth
        self.min_dt_shrinkage = min_dt_shrinkage

        self.linear_combiner_cache = {}

    def get_stability_relevant_init_args(self):
        return (self.use_high_order,)

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def get_linear_combiner(self, arg_count, sample_vec):
        try:
            return self.linear_combiner_cache[arg_count]
        except KeyError:
            lc = self.vector_primitive_factory \
                    .make_linear_combiner(
                            self.dtype, self.scalar_dtype, sample_vec,
                            arg_count=arg_count)
            self.linear_combiner_cache[arg_count] = lc
            return lc


class EmbeddedButcherTableauTimeStepperBase(EmbeddedRungeKuttaTimeStepperBase):
    def __call__(self, y, t, dt, rhs, reject_hook=None):
        from hedge.tools import count_dofs

        # {{{ preparation
        try:
            self.last_rhs
        except AttributeError:
            self.last_rhs = rhs(t, y)
            self.dof_count = count_dofs(self.last_rhs)

            if self.adaptive:
                self.norm = self.vector_primitive_factory \
                        .make_maximum_norm(self.last_rhs)
            else:
                self.norm = None

        # }}}

        flop_count = [0]

        while True:
            rhss = []

            # {{{ stage loop

            for i, (c, coeffs) in enumerate(self.butcher_tableau):
                if len(coeffs) == 0:
                    assert c == 0
                    this_rhs = self.last_rhs
                else:
                    sub_timer = self.timer.start_sub_timer()
                    args = [(1, y)] + [
                            (dt*coeff, rhss[j]) for j, coeff in enumerate(coeffs)
                            if coeff]
                    flop_count[0] += len(args)*2 - 1
                    sub_y = self.limiter(self.get_linear_combiner(
                            len(args), self.last_rhs)(*args))
                    sub_timer.stop().submit()

                    this_rhs = rhs(t + c*dt, sub_y)

                rhss.append(this_rhs)

            # }}}

            def finish_solution(coeffs):
                args = [(1, y)] + [
                        (dt*coeff, rhss[i]) for i, coeff in enumerate(coeffs)
                        if coeff]
                flop_count[0] += len(args)*2 - 1
                return self.get_linear_combiner(
                        len(args), self.last_rhs)(*args)

            if not self.adaptive:
                if self.use_high_order:
                    y = self.limiter(finish_solution(self.high_order_coeffs))
                else:
                    y = self.limiter(finish_solution(self.low_order_coeffs))

                self.last_rhs = this_rhs
                self.flop_counter.add(self.dof_count*flop_count[0])
                return y
            else:
                # {{{ step size adaptation
                high_order_end_y = finish_solution(self.high_order_coeffs)
                low_order_end_y = finish_solution(self.low_order_coeffs)

                flop_count[0] += 3+1  # one two-lincomb, one norm

                # Perform error estimation based on un-limited solutions.
                accept_step, next_dt, rel_err = adapt_step_size(
                        t, dt, y, high_order_end_y, low_order_end_y,
                        self, self.get_linear_combiner(2, high_order_end_y),
                        self.norm)

                if not accept_step:
                    if reject_hook:
                        y = reject_hook(dt, rel_err, t, y)

                    dt = next_dt
                    # ... and go back to top of loop
                else:
                    # finish up
                    self.last_rhs = this_rhs
                    self.flop_counter.add(self.dof_count*flop_count[0])

                    return self.limiter(high_order_end_y), t+dt, dt, next_dt
                # }}}

# }}}


# {{{ Bogacki-Shampine second/third-order Runge-Kutta

class ODE23TimeStepper(EmbeddedButcherTableauTimeStepperBase):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    http://dx.doi.org/10.1016/0893-9659(89)90079-7
    """

    dt_fudge_factor = 1

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

class ODE45TimeStepper(EmbeddedButcherTableauTimeStepperBase):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    http://dx.doi.org/10.1016/0771-050X(80)90013-3.
    """

    dt_fudge_factor = 1

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


# vim: foldmethod=marker
