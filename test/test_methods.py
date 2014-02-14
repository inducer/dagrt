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

from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper, \
                           MultirateFastestFirstEulerMethod
from leap.method.ab import AdamsBashforthTimeStepper
import numpy as np

import logging
logger = logging.getLogger(__name__)


class Basic:
    """
    stolen and modified from ode_systems.py
    ODE-system - basic
    du/dt = v
    dv/dt = -u/t^2
    A = [[0, 1]
        [-1/t^2, 0]].
    """
    def __init__(self):
        self.t_start = 1
        self.t_end = 2
        self.initial_values = np.array([1, 3])

    def f2f_rhs(self, t, u, v):
        return 0

    def s2f_rhs(self, t, u, v):
        return v

    def f2s_rhs(self, t, u, v):
        return -u/t**2

    def s2s_rhs(self, t, u, v):
        return 0

    def soln_0(self, t):
        inner = np.sqrt(3)/2*np.log(t)
        return np.sqrt(t)*(
                5*np.sqrt(3)/3*np.sin(inner)
                + np.cos(inner)
                )

# Run example with
# python test_methods.py "test_multirate_euler_accuracy(MultirateFastestFirstEulerMethod(), 1)"

@pytest.mark.parametrize(("method", "expected_order"),
                         [(MultirateFastestFirstEulerMethod(), 1)])
def test_multirate_euler_accuracy(method, expected_order, show_dag=True, plot_solution=False):
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = "y"
    code = method(component_id)

    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)

    from leap.vm.exec_numpy import NumpyInterpreter, StateComputed

    from ode_systems import Full

    f = Basic()
    soln = f.soln_0

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(4, 7):
        dt = 2**(-n)
        t = 1
        y = np.array([1, 3], dtype=np.float64)
        final_t = 2

        interp = NumpyInterpreter(code,
            rhs_map={"s2f": f.s2f_rhs, "f2f": f.f2f_rhs, "f2s" : f.f2s_rhs,
                     "s2s": f.s2s_rhs})
        interp.set_up(t_start=t, dt_start=dt,
                      state={"fast": y[0], "slow": y[1], "factor": 4})
        interp.initialize()

        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, StateComputed):
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
    #print orderest, order
    assert orderest > expected_order*0.95

# Run example with
# python test_methods.py "test_ab_accuracy(AdamsBashforthTimeStepper(3), 3)"

@pytest.mark.parametrize(("method", "expected_order"), [
    (AdamsBashforthTimeStepper(3), 3),
    ])
def test_ab_accuracy(method, expected_order, show_dag=True, plot_solution=False):
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

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(5, 9):
        dt = 2**(-n)
        t = 1
        y = np.array([1, 3], dtype=np.float64)
        final_t = 10

        interp = NumpyInterpreter(code, rhs_map={component_id: rhs})
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

        error = abs(values[-1]-soln(final_t))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("%s: expected order %d" % (method, expected_order))
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > expected_order*0.95

# }}}

# Run example with
# python test_methods.py "test_rk_accuracy(ODE45TimeStepper(), 5)"

# {{{ non-adaptive test

@pytest.mark.parametrize(("method", "expected_order"), [
    (ODE23TimeStepper(use_high_order=False), 2),
    (ODE23TimeStepper(use_high_order=True), 3),
    (ODE45TimeStepper(use_high_order=False), 4),
    (ODE45TimeStepper(use_high_order=True), 5),
    ])
def test_rk_accuracy(method, expected_order, show_dag=True, plot_solution=False):
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

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(4, 7):
        dt = 2**(-n)
        t = 1
        y = np.array([1, 3], dtype=np.float64)
        final_t = 10

        interp = NumpyInterpreter(code, rhs_map={component_id: rhs})
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

        error = abs(values[-1]-soln(final_t))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("%s: expected order %d" % (method, expected_order))
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    #print orderest, order
    assert orderest > expected_order*0.95

# }}}


# {{{ adaptive test

@pytest.mark.parametrize("method", [
    ODE23TimeStepper(rtol=1e-6),
    ODE45TimeStepper(rtol=1e-6),
    ])
def test_adaptive_timestep(method, show_dag=False, plot=False):
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = "y"
    code = method(component_id)

    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)

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

    from leap.vm.exec_numpy import (
            NumpyInterpreter,
            StateComputed, StepCompleted, StepFailed)

    interp = NumpyInterpreter(code, rhs_map={component_id: example})
    interp.set_up(t_start=example.t_start, dt_start=1e-5, state={component_id: y})
    interp.initialize()

    times = []
    values = []

    new_times = []
    new_values = []

    last_t = 0
    step_sizes = []

    for event in interp.run(t_end=example.t_end):
        if isinstance(event, StateComputed):
            assert event.component_id == component_id
            assert event.t < example.t_end + 1e-12

            new_values.append(event.state_component)
            new_times.append(event.t)
        elif isinstance(event, StepCompleted):
            step_sizes.append(event.t - last_t)
            last_t = event.t

            times.extend(new_times)
            values.extend(new_values)
            del new_times[:]
            del new_values[:]
        elif isinstance(event, StepFailed):
            del new_times[:]
            del new_values[:]

            logger.info("failed step at t=%s" % event.t)

        #if step % 100 == 0:
            #print t

    times = np.array(times)
    values = np.array(values)

    if plot:
        import matplotlib.pyplot as pt
        pt.plot(times, values[:, 1], "x-")
        pt.show()
        pt.plot(times, step_sizes, "x-")
        pt.show()

    step_sizes = np.array(step_sizes)
    small_step_frac = len(np.nonzero(step_sizes < 0.01)[0]) / len(step_sizes)
    big_step_frac = len(np.nonzero(step_sizes > 0.05)[0]) / len(step_sizes)

    print(small_step_frac)
    print(big_step_frac)
    assert small_step_frac <= 0.35
    assert big_step_frac >= 0.16

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
