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

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


# {{{ non-adaptive test

# test using 'python test_rk.py test_rk_accuracy(pmi_int, ODE23TimeStepper(use_high_order=False), 2)'  # noqa

@pytest.mark.parametrize(("method", "expected_order"), [
    (ODE23TimeStepper("y", use_high_order=False), 2),
    (ODE23TimeStepper("y", use_high_order=True), 3),
    (ODE45TimeStepper("y", use_high_order=False), 4),
    (ODE45TimeStepper("y", use_high_order=True), 5),
    ])
def test_rk_accuracy(python_method_impl, method, expected_order,
                     show_dag=False, plot_solution=False):
    from utils import check_simple_convergence
    check_simple_convergence(method=method, method_impl=python_method_impl,
                             expected_order=expected_order, show_dag=show_dag,
                             plot_solution=plot_solution)

# }}}


# {{{ adaptive test

@pytest.mark.parametrize("method", [
    ODE23TimeStepper("y", rtol=1e-6),
    ODE45TimeStepper("y", rtol=1e-6),
    ])
def test_adaptive_timestep(python_method_impl, method, show_dag=False,
                           plot=False):
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = method.component_id
    code = method.generate()

    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)

    from stiff_test_systems import VanDerPolProblem
    example = VanDerPolProblem()
    y = example.initial()

    interp = python_method_impl(code,
                                function_map={"<func>" + component_id: example})
    interp.set_up(t_start=example.t_start, dt_start=1e-5, context={component_id: y})

    times = []
    values = []

    new_times = []
    new_values = []

    last_t = 0
    step_sizes = []

    for event in interp.run(t_end=example.t_end):
        if isinstance(event, interp.StateComputed):
            assert event.component_id == component_id

            new_values.append(event.state_component)
            new_times.append(event.t)
        elif isinstance(event, interp.StepCompleted):
            step_sizes.append(event.t - last_t)
            last_t = event.t

            times.extend(new_times)
            values.extend(new_values)
            del new_times[:]
            del new_values[:]
        elif isinstance(event, interp.StepFailed):
            del new_times[:]
            del new_values[:]

            logger.info("failed step at t=%s" % event.t)

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
