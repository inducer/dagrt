# -*- coding: utf-8 -*-
# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
# Copyright (C) 2014 Matt Wala
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import numpy
import numpy.linalg as la
import pytest
from pytools import memoize_method


class MultirateTimesteperAccuracyChecker:
    """Check that the multirate timestepper has the advertised accuracy."""

    def __init__(self, method, order, step_ratio, ode, display_dag=False,
                 display_solution=False):
        self.method = method
        self.order = order
        self.step_ratio = step_ratio
        self.ode = ode
        self.display_dag = display_dag
        self.display_solution = display_solution

    @memoize_method
    def get_code(self):
        from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
        from pytools import DictionaryWithDefault        
        order = DictionaryWithDefault(lambda x : self.order)
        stepper = TwoRateAdamsBashforthTimeStepper(self.method, order,
                                                       self.step_ratio)
        return stepper()
    
    def initialize_interpreter(self, dt):
        from leap.vm.exec_numpy import NumpyInterpreter

        # Requires a coupled component.
        def make_coupled(f2f, f2s, s2f, s2s):
            def coupled(t, y):
                args = (t, y[0] + y[1], y[2] + y[3])
                return numpy.array((f2f(*args), f2s(*args), s2f(*args),
                    s2s(*args)),)
            return coupled
        
        rhs_map = { '<func>f2f' : self.ode.f2f_rhs,
            '<func>s2f' : self.ode.s2f_rhs, '<func>f2s' : self.ode.f2s_rhs,
            '<func>s2s' : self.ode.s2s_rhs, '<func>coupled' : make_coupled(
            self.ode.f2f_rhs, self.ode.s2f_rhs, self.ode.f2s_rhs,
            self.ode.s2s_rhs) }
        interpreter = NumpyInterpreter(self.get_code(), rhs_map)
        
        t = self.ode.t_start
        y = self.ode.initial_values
        interpreter.set_up(t_start=t, dt_start=dt, state = {'fast': y[0],
            'slow' : y[1]})
        interpreter.initialize()
        return interpreter

    def get_error(self, dt, name=None, plot_solution=False):
        from leap.vm.exec_numpy import StateComputed
        
        final_t = self.ode.t_end
        
        interp = self.initialize_interpreter(dt)
        
        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, StateComputed):
                values.append(event.state_component)
                times.append(event.t)
        
        assert abs(times[-1] - final_t) < 1e-10
                
        t = times[-1]
        y = values[-1]
        
        from ode_systems import Basic, Tria
        
        proj = lambda l, x: [z[x] for z in l]

        if isinstance(self.ode, Basic) or isinstance(self.ode, Tria):
            # AK: why?
            if self.display_solution:
                self.plot_solution(times, proj(values, 0), self.ode.soln_0)
            return abs(y[0]-self.ode.soln_0(t))
        else:
            from math import sqrt
            if self.display_solution:
                self.plot_solution(times, proj(values, 0), self.ode.soln_0)
                self.plot_solution(times, proj(values, 1), self.ode.soln_1)
            return abs(sqrt(y[0]**2 + y[1]**2)
                    - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2)
                    )

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
        for n in range(5,8):
            dt = 2**(-n)
            error = self.get_error(dt, "mrab-%d.dat" % self.order)
            eocrec.add_data_point(dt, error)

        print("------------------------------------------------------")
        print("ORDER %d" % self.order)
        print("------------------------------------------------------")
        print(eocrec.pretty_print())

        orderest = eocrec.estimate_order_of_convergence()[0,1]
        assert orderest > self.order*0.70

# Run example with
# python test_multirate.py "test_multirate_accuracy(\"F\", 3)"
@pytest.mark.parametrize(("method", "expected_order"), [
    ("F", 3),
    ("Fq", 3)
    ])
def test_multirate_accuracy(method, expected_order, show_dag=False,
                            plot_solution=False):
    from ode_systems import Full
    from leap.method.ab.multirate.methods import methods
    m = methods[method]
    
    checker = MultirateTimesteperAccuracyChecker(m, expected_order, 2, Full(),
        show_dag, plot_solution)
    checker()


@pytest.mark.slowtest
def test_all_multirate_accuracy():
    """Check that the multirate timestepper has the advertised accuracy"""

    from leap.method.ab.multirate.methods import methods
    import ode_systems

    step_ratio = 2

    for sys_name in ["Basic", "Full", "Comp", "Tria"]:
        system = getattr(ode_systems, sys_name)

        for name in methods:
            print("------------------------------------------------------")
            print("METHOD: %s" % name)
            print("------------------------------------------------------")
            for order in [1, 3, 5]:
                MultirateTimesteperAccuracyChecker(
                        methods[name], order, step_ratio, ode = system())()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

