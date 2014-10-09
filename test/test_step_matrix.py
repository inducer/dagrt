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

from leap.method.ab import AdamsBashforthTimeStepper
import numpy as np

import logging
logger = logging.getLogger(__name__)


# Run example with
# python test_step_matrix.py "test_step_matrix()"

def test_step_matrix(show_matrix=True, show_dag=False):
    component_id = 'y'
    code = AdamsBashforthTimeStepper(3)(component_id)
    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)
    from leap.vm.exec_numpy import StepMatrixFinder

    def rhs(t, y):
        return -y

    def rhs_deriv(n, t, y):
        return -1

    y = 1.0
    t = 0.0
    final_t = 1.0
    dt = 0.1

    step_matrices = []
    finder = StepMatrixFinder(code, rhs_map={component_id: rhs},
       rhs_deriv_map={component_id: rhs_deriv})
    finder.set_up(t_start=t, dt_start=dt, state={component_id: y})
    finder.initialize()
    for event in finder.run(t_end=final_t):
        if isinstance(event, finder.StateComputed):
            step_matrices.append(event.step_matrix)
    if show_matrix:
        print('Variables: %s' % [var.name for var in finder.get_state_variables()])
        np.set_printoptions(precision=2)
        print(step_matrices[-2])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
