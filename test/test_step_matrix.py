#! /usr/bin/env python

from __future__ import division, with_statement, print_function

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
import pytest

from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper
import numpy as np
import numpy.linalg as la

import logging
logger = logging.getLogger(__name__)


# Run example with
# python test_step_matrix.py "test_step_matrix(ODE23TimeStepper())"

@pytest.mark.parametrize("method", [
    ODE23TimeStepper(use_high_order=False),
    ODE23TimeStepper(use_high_order=True),
    ODE45TimeStepper(use_high_order=False),
    ODE45TimeStepper(use_high_order=True),
    ])
def test_step_matrix(method, show_matrix=True, show_dag=False):
    component_id = 'y'
    code = method(component_id)
    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)
    from leap.vm.exec_numpy import StepMatrixFinder, NumpyInterpreter

    from pymbolic import var

    # {{{ build matrix

    def rhs_sym(t, y):
        return var("lambda")*y

    finder = StepMatrixFinder(code, function_map={"<func>" + component_id: rhs_sym})

    mat = finder.get_state_step_matrix("primary")

    if show_matrix:
        print('Variables: %s' % finder.variables)
        from pytools import indices_in_shape
        for i in indices_in_shape(mat.shape):
            print(i, mat[i])

    # }}}

    dt = 0.1
    lambda_ = -0.4

    def rhs(t, y):
        return lambda_*y

    interp = NumpyInterpreter(code, function_map={"<func>" + component_id: rhs})
    interp.set_up(t_start=0, dt_start=dt, context={component_id: 15})

    assert interp.next_state == "initialization"
    for event in interp.run_single_step():
        pass
    assert interp.next_state == "primary"

    start_values = np.array(
            [interp.context[v] for v in finder.variables])

    for event in interp.run_single_step():
        pass
    assert interp.next_state == "primary"

    stop_values = np.array(
            [interp.context[v] for v in finder.variables])

    from leap.vm.expression import EvaluationMapper
    concrete_mat = EvaluationMapper({
        "lambda": lambda_,
        "<dt>": dt,
        }, {})(mat)

    stop_values_from_mat = concrete_mat.dot(start_values)

    rel_err = (
            la.norm(stop_values - stop_values_from_mat)
            /
            la.norm(stop_values))

    assert rel_err < 1e-12


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
