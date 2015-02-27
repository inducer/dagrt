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

from leap.vm.language import AssignExpression, YieldState
from leap.vm.language import CodeBuilder, TimeIntegratorCode
from pymbolic import var

from utils import execute_and_return_single_result


@pytest.mark.parametrize(('obj, len_'), [(np.ones(0), 0),
                                         (np.ones(1), 1),
                                         (np.ones(2), 2),
                                         (6.0, 1)])
def test_len(python_method_impl, obj, len_):
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_1', assignee='x',
                         expression=var('<builtin>len')(obj)),
        YieldState(id='return', time=0, time_id='final',
                   expression=var('x'), component_id='<state>',
                   depends_on=['assign_1']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_steady_state(
            dep_on=['return'],
            instructions=cbuild.instructions)

    result = execute_and_return_single_result(python_method_impl, code)
    assert result == len_


@pytest.mark.parametrize(('value'), [0, float('nan')])
def test_isnan(python_method_impl, value):
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_1', assignee='x',
                         expression=var('<builtin>isnan')(value)),
        YieldState(id='return', time=0, time_id='final',
                   expression=var('x'), component_id='<state>',
                   depends_on=['assign_1']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_steady_state(
            dep_on=['return'],
            instructions=cbuild.instructions)

    result = execute_and_return_single_result(python_method_impl, code)
    assert result == np.isnan(value)


@pytest.mark.parametrize(('order'), [2, np.inf])
@pytest.mark.parametrize(('test_vector'),
                         [6, 1j, np.array([-3]), np.array([-3, 4])])
def test_norm(python_method_impl, order, test_vector):

    def true_norm(x):
        if np.isscalar(x):
            return abs(x)
        return np.linalg.norm(x, ord=order)

    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_1', assignee='x',
                         expression=test_vector),
        AssignExpression(id='assign_2', assignee='n',
                         expression=var('<builtin>norm')(var('x'),
                                                         ord=order),
                         depends_on=['assign_1']),
        YieldState(id='return', time=0, time_id='final',
                   expression=var('n'), component_id='<state>',
                   depends_on=['assign_2']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_steady_state(
            dep_on=['return'],
            instructions=cbuild.instructions)

    result = execute_and_return_single_result(python_method_impl, code)
    assert np.allclose(result, true_norm(test_vector))


@pytest.mark.parametrize(('x, y'), [(1.0, 1.0j), (1.0j, 1.0),
                                    (1.0, 1.0), (1.0j, 1.0j)])
def test_dot_product(python_method_impl, x, y):
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_1', assignee='x',
                         expression=var('<builtin>dot_product')(x, y)),
        YieldState(id='return', time=0, time_id='final',
                   expression=var('x'), component_id='<state>',
                   depends_on=['assign_1']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_steady_state(
            dep_on=['return'],
            instructions=cbuild.instructions)

    result = execute_and_return_single_result(python_method_impl, code)
    assert result == np.vdot(x, y)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
