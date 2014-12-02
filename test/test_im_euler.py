#! /usr/bin/env python

from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner, Matt Wala"

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

import sys

from leap.method.im_euler import ImplicitEulerMethod
from leap.vm.implicit import GenericNumpySolver
import numpy as np
import scipy.optimize


class ScipyRootSolver(GenericNumpySolver):

    def run_solver(self, func, guess):
        return scipy.optimize.root(func, guess).x


def test_im_euler_accuracy(show_dag=False, plot_solution=False):
    component_id = "y"
    method = ImplicitEulerMethod()
    code = method(component_id)

    expected_order = 1

    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)

    from leap.vm.exec_numpy import NumpyInterpreter

    h = -0.5
    y_0 = 1.0

    def rhs(t, y):
        return h * y

    def soln(t):
        return y_0 * np.exp(h * t)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(1, 5):
        dt = 2**(-n)
        t = 0.0
        y = y_0
        final_t = 1

        interp = NumpyInterpreter(code, function_map={component_id: rhs},
                                  solver_map={'newton': ScipyRootSolver()})
        interp.set_up(t_start=t, dt_start=dt, context={component_id: y})
        interp.initialize()

        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, interp.StateComputed):
                assert event.component_id == component_id
                values.append(event.state_component)
                times.append(event.t)

        assert abs(times[-1] - final_t) < 1e-10

        times = np.array(times)

        if plot_solution:
            import matplotlib.pyplot as pt
            pt.plot(times, values, label="comp")
            pt.plot(times, soln(times), label="true")
            pt.show()

        error = abs(values[-1]-soln(final_t))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("%s: expected order %d" % (method, expected_order))
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > expected_order*0.9


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
