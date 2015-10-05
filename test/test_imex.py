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

from leap.method.rk.imex import KennedyCarpenterIMEXARK4Method
from stiff_test_systems import KapsProblem
from leap.vm.implicit import ScipySolverGenerator

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


_component_id = "y"


@pytest.mark.parametrize("problem, method, expected_order", [
    [KapsProblem(epsilon=0.9), KennedyCarpenterIMEXARK4Method(_component_id,
                                                        use_high_order=False), 3],
    [KapsProblem(epsilon=0.9), KennedyCarpenterIMEXARK4Method(_component_id), 4],
    ])
def test_convergence(python_method_impl, problem, method, expected_order):
    pytest.importorskip("scipy")

    code = method.generate()

    sgen = ScipySolverGenerator(*method.implicit_expression())
    from leap.vm.implicit import replace_AssignSolved
    code = replace_AssignSolved(code, {"solve": sgen})

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(2, 7):
        dt = 2**(-n)

        y_0 = problem.initial()
        t_start = problem.t_start
        t_end = problem.t_end

        interp = python_method_impl(code, function_map={
            "<func>expl_y": problem.nonstiff,
            "<func>impl_y": problem.stiff,
            sgen.solver_func.name: sgen.get_compiled_solver(),
        })

        interp.set_up(t_start=t_start, dt_start=dt, context={_component_id: y_0})

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


@pytest.mark.parametrize("problem, method", [
    [KapsProblem(epsilon=0.001), KennedyCarpenterIMEXARK4Method],
    ])
def test_adaptive(python_method_impl, problem, method):
    pytest.importorskip("scipy")

    t_start = problem.t_start
    t_end = problem.t_end
    dt = 1.0e-1

    tols = [10.0 ** (-j) for j in range(1, 5)]

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    # Test that tightening the tolerance will decrease the overall error.
    for atol in tols:
        generator = method(_component_id, atol=atol)
        code = generator.generate()
        print(code)

        sgen = ScipySolverGenerator(*generator.implicit_expression())
        from leap.vm.implicit import replace_AssignSolved
        code = replace_AssignSolved(code, {"solve": sgen})

        interp = python_method_impl(code, function_map={
            "<func>expl_y": problem.nonstiff,
            "<func>impl_y": problem.stiff,
            sgen.solver_func.name: sgen.get_compiled_solver()
        })
        interp.set_up(t_start=t_start, dt_start=dt,
                      context={_component_id: problem.initial()})

        times = []
        values = []

        new_times = []
        new_values = []

        for event in interp.run(t_end=t_end):
            clear_flag = False
            if isinstance(event, interp.StateComputed):
                assert event.component_id == _component_id
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
        exact = problem.exact(times[-1])
        error = np.linalg.norm(values[-1] - exact)
        eocrec.add_data_point(atol, error)

    print("Error vs. tolerance")
    print(eocrec.pretty_print())
    order = eocrec.estimate_order_of_convergence()[0, 1]
    assert order > 0.9


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
