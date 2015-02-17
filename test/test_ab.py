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
import pytest
from leap.method.ab import AdamsBashforthTimeStepper

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_ab_accuracy(python_method_impl, order, show_dag=False,
                     plot_solution=False):
    from utils import check_simple_convergence
    method = AdamsBashforthTimeStepper("y", order)
    check_simple_convergence(method=method, method_impl=python_method_impl,
                             expected_order=order, show_dag=show_dag,
                             plot_solution=plot_solution)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
