# -*- coding: utf-8 -*-

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
import pytest
from pytools import memoize_method

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


class MultirateTimestepperAccuracyChecker(object):
    """Check that the multirate timestepper has the advertised accuracy."""

    def __init__(self, method, order, step_ratio, ode, method_impl,
                 display_dag=False, display_solution=False):
        self.method = method
        self.order = order
        self.step_ratio = step_ratio
        self.ode = ode
        self.method_impl = method_impl
        self.display_dag = display_dag
        self.display_solution = display_solution

    @memoize_method
    def get_code(self):
        from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
        from pytools import DictionaryWithDefault
        order = DictionaryWithDefault(lambda x: self.order)
        stepper = TwoRateAdamsBashforthTimeStepper(self.method, order,
                                                       self.step_ratio)
        return stepper()

    def initialize_method(self, dt):
        # Requires a coupled component.
        def make_coupled(f2f, f2s, s2f, s2s):
            def coupled(t, y):
                args = (t, y[0] + y[1], y[2] + y[3])
                return numpy.array((f2f(*args), f2s(*args), s2f(*args),
                    s2s(*args)),)
            return coupled

        function_map = {'<func>f2f': self.ode.f2f_rhs,
            '<func>s2f': self.ode.s2f_rhs, '<func>f2s': self.ode.f2s_rhs,
            '<func>s2s': self.ode.s2s_rhs, '<func>coupled': make_coupled(
                self.ode.f2f_rhs, self.ode.s2f_rhs, self.ode.f2s_rhs,
                self.ode.s2s_rhs)}

        method = self.method_impl(self.get_code(), function_map=function_map)

        t = self.ode.t_start
        y = self.ode.initial_values
        method.set_up(t_start=t, dt_start=dt,
                context={'fast': y[0], 'slow': y[1]})
        method.initialize()
        return method

    def get_error(self, dt, name=None, plot_solution=False):
        final_t = self.ode.t_end

        method = self.initialize_method(dt)

        times = []
        slow = []
        fast = []
        for event in method.run(t_end=final_t):
            if isinstance(event, method.StateComputed):
                if event.component_id == "slow":
                    slow.append(event.state_component)
                    times.append(event.t)
                elif event.component_id == "fast":
                    fast.append(event.state_component)

        assert abs(times[-1] - final_t) < 1e-10

        t = times[-1]
        y = (fast[-1], slow[-1])

        from ode_systems import Basic, Tria

        if isinstance(self.ode, Basic) or isinstance(self.ode, Tria):
            # AK: why?
            if self.display_solution:
                self.plot_solution(times, fast, self.ode.soln_0)
            return abs(y[0]-self.ode.soln_0(t))
        else:
            from math import sqrt
            if self.display_solution:
                self.plot_solution(times, fast, self.ode.soln_0)
                self.plot_solution(times, slow, self.ode.soln_1)
            return abs(sqrt(y[0]**2 + y[1]**2)
                    - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2))

    def show_dag(self):
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(self.get_code())

    def plot_solution(self, times, values, soln, label=None):
        import matplotlib.pyplot as pt
        pt.plot(times, values, label="comp")
        pt.plot(times, soln(times), label="true")
        pt.legend(loc='best')
        pt.show()

    def __call__(self):
        """Run the test and output the estimated the order of convergence."""

        from pytools.convergence import EOCRecorder

        if self.display_dag:
            self.show_dag()

        eocrec = EOCRecorder()
        for n in range(6, 8):
            dt = 2**(-n)
            error = self.get_error(dt, "mrab-%d.dat" % self.order)
            eocrec.add_data_point(dt, error)

        print("------------------------------------------------------")
        print("ORDER %d" % self.order)
        print("------------------------------------------------------")
        print(eocrec.pretty_print())

        orderest = eocrec.estimate_order_of_convergence()[0, 1]
        assert orderest > self.order*0.70


@pytest.mark.slowtest
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("system", ["Basic", "Full", "Comp", "Tria"])
def test_multirate_accuracy(python_method_impl, order, system):
    """Check that the multirate timestepper has the advertised accuracy"""

    from leap.method.ab.multirate.methods import methods
    import ode_systems

    step_ratio = 2

    system = getattr(ode_systems, system)

    for name in methods:
        print("------------------------------------------------------")
        print("METHOD: %s" % name)
        print("------------------------------------------------------")
        MultirateTimestepperAccuracyChecker(
            methods[name], order, step_ratio, ode=system(),
            method_impl=python_method_impl)()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
