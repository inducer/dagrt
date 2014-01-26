#! /usr/bin/env python

from __future__ import division, with_statement

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

import sys
import pytest

from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper
import numpy as np

import logging
logger = logging.getLogger(__name__)


# Run example with
# python test_methods.py "test_rk_accuracy(ODE45TimeStepper(), 5)"

@pytest.mark.parametrize(("method", "expected_order"), [
    (ODE23TimeStepper(use_high_order=False), 2),
    (ODE23TimeStepper(use_high_order=True), 3),
    (ODE45TimeStepper(use_high_order=False), 4),
    (ODE45TimeStepper(use_high_order=True), 5),
    ])
def test_rk_accuracy(method, expected_order, show_dag=False, plot_solution=False):
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = "y"
    code = method(component_id)

    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)

    from leap.vm.exec_numpy import NumpyInterpreter, StateComputed

    def rhs(t, y):
        u, v = y
        return np.array([v, -u/t**2], dtype=np.float64)

    def soln(t):
        inner = np.sqrt(3)/2*np.log(t)
        return np.sqrt(t)*(
                5*np.sqrt(3)/3*np.sin(inner)
                + np.cos(inner)
                )

    def get_error(interp, dt):
        t = 1
        y = np.array([1, 3], dtype=np.float64)
        final_t = 10

        interp.set_up(t_start=t, dt_start=dt, state={component_id: y})
        interp.initialize()

        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, StateComputed):
                assert event.component_id == component_id
                values.append(event.state_component[0])
                times.append(event.t)

        assert abs(times[-1] - final_t) < 1e-10

        times = np.array(times)

        if plot_solution:
            import matplotlib.pyplot as pt
            pt.plot(times, values, label="comp")
            pt.plot(times, soln(times), label="true")
            pt.show()

        return abs(values[-1]-soln(final_t))

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(4, 7):
        dt = 2**(-n)
        interp = NumpyInterpreter(code, rhs_map={component_id: rhs})

        error = get_error(interp, dt)
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("%s: expected order %d" % (method, expected_order))
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    #print orderest, order
    assert orderest > expected_order*0.95


def no_test_adaptive_timestep():
    class VanDerPolOscillator:
        def __init__(self, mu=30):
            self.mu = mu
            self.t_start = 0
            self.t_end = 100

        def ic(self):
            return np.array([2, 0], dtype=np.float64)

        def __call__(self, t, y):
            u1 = y[0]
            u2 = y[1]
            return np.array([
                u2,
                -self.mu*(u1**2-1)*u2-u1],
                dtype=np.float64)

    example = VanDerPolOscillator()
    y = example.ic()

    from hedge.timestep.dumka3 import Dumka3TimeStepper
    stepper = Dumka3TimeStepper(3, rtol=1e-6)

    next_dt = 1e-5
    from hedge.timestep import times_and_steps
    times = []
    hist = []
    dts = []
    for step, t, max_dt in times_and_steps(
            max_dt_getter=lambda t: next_dt,
            taken_dt_getter=lambda: taken_dt,
            start_time=example.t_start, final_time=example.t_end):

        #if step % 100 == 0:
            #print t

        hist.append(y)
        times.append(t)
        y, t, taken_dt, next_dt = stepper(y, t, next_dt, example)
        dts.append(taken_dt)

    if False:
        from matplotlib.pyplot import plot, show
        plot(times, [h_entry[1] for h_entry in hist])
        show()
        plot(times, dts)
        show()

    dts = np.array(dts)
    small_step_frac = len(np.nonzero(dts < 0.01)[0]) / step
    big_step_frac = len(np.nonzero(dts > 0.1)[0]) / step
    assert abs(small_step_frac - 0.6) < 0.1
    assert abs(big_step_frac - 0.2) < 0.1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
