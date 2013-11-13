"""SSP Runge-Kutta methods."""

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

from leap.method.rk import EmbeddedRungeKuttaMethod, adapt_step_size


# {{{ Shu-Osher-form SSP RK

class EmbeddedShuOsherFormTimeStepperBase(EmbeddedRungeKuttaMethod):
    r"""
    The attribute *shu_osher_tableau* is defined by and consists of a tuple of
    lists of :math:`\alpha` and :math:`\beta` as given in (2.10) of [1]. Each
    list entry contains a coefficient and an index. Within the list of
    :math:`\alpha`, these index into values of :math:`u^{(i)}`, where the
    initial condition has index 0, the first row of the tableau index 1, and so
    on.  Within the list of :math:`beta`, the index is into function
    evaluations at :math:`u^{(i)}`.

    *low_order_index* and *high_order_index* give the result of the embedded
    high- and low-order methods.

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    def __call__(self, y, t, dt, rhs, reject_hook=None):

        flop_count = 0

        def get_rhs(i):
            try:
                return rhss[i]
            except KeyError:
                result = rhs(t + time_fractions[i]*dt, row_values[i])
                rhss[i] = result

                try:
                    self.dof_count
                except AttributeError:
                    from hedge.tools import count_dofs
                    self.dof_count = count_dofs(result)

                return result

        while True:
            time_fractions = [0]
            row_values = [y]
            rhss = {}

            # {{{ row loop

            for alpha_list, beta_list in self.shu_osher_tableau:
                sub_timer = self.timer.start_sub_timer()
                args = ([(alpha, row_values[i]) for alpha, i in alpha_list]
                        + [(dt*beta, get_rhs(i)) for beta, i in beta_list])
                flop_count += len(args)*2 - 1

                some_rhs = iter(rhss.itervalues()).next()
                row_values.append(
                        self.limiter(
                            self.get_linear_combiner(len(args), some_rhs)(*args)))
                sub_timer.stop().submit()

                time_fractions.append(
                        sum(alpha * time_fractions[i] for alpha, i in alpha_list)
                        + sum(beta for beta, i in beta_list))

            # }}}

            if not self.adaptive:
                self.flop_counter.add(self.dof_count*flop_count)

                if self.use_high_order:
                    assert abs(time_fractions[self.high_order_index] - 1) < 1e-15
                    return row_values[self.high_order_index]
                else:
                    assert abs(time_fractions[self.low_order_index] - 1) < 1e-15
                    return row_values[self.low_order_index]
            else:
                # {{{ step size adaptation

                assert abs(time_fractions[self.high_order_index] - 1) < 1e-15
                assert abs(time_fractions[self.low_order_index] - 1) < 1e-15

                high_order_end_y = row_values[self.high_order_index]
                low_order_end_y = row_values[self.low_order_index]

                some_rhs = iter(rhss.itervalues()).next()

                try:
                    norm = self.norm
                except AttributeError:
                    norm = self.norm = self.vector_primitive_factory \
                            .make_maximum_norm(some_rhs)

                flop_count += 3+1  # one two-lincomb, one norm
                accept_step, next_dt, rel_err = adapt_step_size(
                        t, dt, y, high_order_end_y, low_order_end_y,
                        self, self.get_linear_combiner(2, some_rhs), norm)

                if not accept_step:
                    if reject_hook:
                        y = reject_hook(dt, rel_err, t, y)

                    dt = next_dt
                    # ... and go back to top of loop
                else:
                    # finish up
                    self.flop_counter.add(self.dof_count*flop_count)

                    return high_order_end_y, t+dt, dt, next_dt

                # }}}


class SSP2TimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Theorem 2.2 of Section 2.4.1 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    shu_osher_tableau = [
            ([(1, 0)], [(1, 0)]),
            ([(1/2, 0), (1/2, 1)], [(1/2, 1)]),
            ]

    # no low-order
    high_order = 2
    high_order_index = 2


class SSP3TimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Theorem 2.2 of Section 2.4.1 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    shu_osher_tableau = [
            ([(1, 0)], [(1, 0)]),
            ([(3/4, 0), (1/4, 1)], [(1/4, 1)]),
            ([(1/3, 0), (2/3, 2)], [(2/3, 2)]),
            ]

    # no low-order
    high_order = 3
    high_order_index = 3


class SSP23FewStageTimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Example 6.1 of Section 6.3 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    shu_osher_tableau = [
            ([(1, 0)], [(1/2, 0)]),
            ([(1, 1)], [(1/2, 1)]),
            ([(1/3, 0), (2/3, 2)], [(1/2*2/3, 2)]),
            ([(2/3, 0), (1/3, 2)], [(1/2*1/3, 2)]),
            ([(1, 4)], [(1/2, 4)]),
            ]

    low_order = 2
    low_order_index = 3
    high_order = 3
    high_order_index = 5


class SSP23ManyStageTimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Example 6.2 of Section 6.3 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    # This has a known bug--perhaps an issue with reference [1]?
    # Entry 7 is not consistent as a second-order approximation.

    shu_osher_tableau = [
                ([(1, i)], [(1/6, i)]) for i in range(6)
            ]+[
                ([(1/7, 0), (6*1/7, 6)], [(1/6*1/7, 6)]),  # 7
                ([(3/5, 1), (2/5, 6)], []),  # 8: u^{(6)\ast}
            ]+[
                ([(1, i-1)], [(1/6, i-1)]) for i in range(9, 9+2+1)
            ]

    low_order = 2
    low_order_index = 7
    high_order = 3
    high_order_index = -1

# }}}


# vim: foldmethod=marker
