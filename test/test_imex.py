#! /usr/bin/env python

from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2014 Matt Wala"

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
import pytest
import sys

from leap.method.rk.imex import KennedyCarpenterIMEXARK4
from stiff_test_systems import KapsProblem, VanDerPolProblem
from leap.vm.implicit import GenericNumpySolver
import scipy.optimize


class ScipyRootSolver(GenericNumpySolver):

    def run_solver(self, func, guess):
        return scipy.optimize.root(func, guess, method='broyden1').x


@pytest.mark.parametrize("problem, method, expected_order", [
    [KapsProblem(epsilon=0.5), KennedyCarpenterIMEXARK4(), 4]
    ])
def test_convergence(problem, method, expected_order):
    component_id = "y"
    code = method(component_id, solver_id='solver')

    from leap.vm.exec_numpy import NumpyInterpreter
    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(1, 7):
        dt = 2**(-n)

        y_0 = problem.initial()
        t_start = problem.t_start
        t_end = problem.t_end

        interp = NumpyInterpreter(code, function_map={
                '<func>expl_' + component_id: problem.nonstiff,
                '<func>impl_' + component_id: problem.stiff
                },
                solver_map={'solver': ScipyRootSolver()})
        interp.set_up(t_start=t_start, dt_start=dt, context={component_id: y_0})
        interp.initialize()

        times = []
        values = []

        for event in interp.run(t_end=t_end):
            if isinstance(event, interp.StateComputed):
                values.append(event.state_component)
                times.append(event.t)

        times = np.array(times)
        values = np.array(values)

        assert abs(times[-1] - t_end) < 1e-10

        times = np.array(times)

        error = np.linalg.norm(values[-1] - problem.exact(t_end))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("expected order %d" % expected_order)
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > 0.9 * expected_order


@pytest.mark.parametrize("problem, method, rel_error_bound", [
    [KapsProblem(epsilon=0.001), KennedyCarpenterIMEXARK4(rtol=1.0e-6), 1.0e-6],
    [VanDerPolProblem(), KennedyCarpenterIMEXARK4(rtol=1.0e-6), None]
    ])
def test_kaps_problem(problem, method, rel_error_bound):
    from leap.vm.exec_numpy import NumpyInterpreter

    component_id = 'y'

    y_0 = problem.initial()
    t_start = problem.t_start
    t_end = problem.t_end
    dt = 1.0e-2

    code = method(component_id, solver_id='solver')

    interp = NumpyInterpreter(code, function_map={
            '<func>expl_' + component_id: problem.nonstiff,
            '<func>impl_' + component_id: problem.stiff
            },
            solver_map={'solver': ScipyRootSolver()})
    interp.set_up(t_start=t_start, dt_start=dt, context={component_id: y_0})
    interp.initialize()

    times = []
    values = []

    new_times = []
    new_values = []

    for event in interp.run(t_end=t_end):
        clear_flag = False
        if isinstance(event, interp.StateComputed):
            assert event.component_id == component_id
            new_values.append(event.state_component)
            new_times.append(event.t)
        elif isinstance(event, interp.StepCompleted):
            values.extend(new_values)
            times.extend(new_times)
            clear_flag = True
        elif isinstance(event, interp.StepFailed):
            clear_flag = True
        if clear_flag:
            del new_times[:]
            del new_values[:]

    times = np.array(times)
    values = np.array(values)
    assert abs(times[-1] - t_end) < 1e-10
    try:
        exact = problem.exact(times[-1])
        rel_error = np.linalg.norm(values[-1] - exact) / np.linalg.norm(exact)
        assert rel_error < rel_error_bound
    except NotImplementedError:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
