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


class BuiltinsTestBase(object):
    """A collection of tests for builtin functions."""

    def execute_and_return_single_result(self, code):
        raise NotImplementedError()

    @pytest.mark.parametrize(('len_'), [0, 1, 2])
    def test_len(self, len_):
        test_vector = np.ones(len_)
        cbuild = CodeBuilder()
        cbuild.add_and_get_ids(
            AssignExpression(id='assign_1', assignee='x',
                             expression=var('<builtin>len')(test_vector)),
            YieldState(id='return', time=0, time_id='final',
                        expression=var('x'), component_id='<state>',
                        depends_on=['assign_1']))
        cbuild.commit()
        code = TimeIntegratorCode(initialization_dep_on=[],
                                  instructions=cbuild.instructions,
                                  step_dep_on=['return'],
                                  step_before_fail=False)
        result = self.execute_and_return_single_result(code)
        assert result == len(test_vector)

    @pytest.mark.parametrize(('value'), [0, float('nan')])
    def test_isnan(self, value):
        cbuild = CodeBuilder()
        cbuild.add_and_get_ids(
            AssignExpression(id='assign_1', assignee='x',
                             expression=var('<builtin>isnan')(value)),
            YieldState(id='return', time=0, time_id='final',
                        expression=var('x'), component_id='<state>',
                        depends_on=['assign_1']))
        cbuild.commit()
        code = TimeIntegratorCode(initialization_dep_on=[],
                                  instructions=cbuild.instructions,
                                  step_dep_on=['return'],
                                  step_before_fail=False)
        result = self.execute_and_return_single_result(code)
        assert result == np.isnan(value)

    @pytest.mark.parametrize(('order'), [2, np.inf])
    def test_norm(self, order):
        test_vector = np.array([-3, 4], dtype=np.double)
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
        code = TimeIntegratorCode(initialization_dep_on=[],
                                  instructions=cbuild.instructions,
                                  step_dep_on=['return'],
                                  step_before_fail=False)
        result = self.execute_and_return_single_result(code)
        expected_result = np.linalg.norm(test_vector, ord=order)
        assert np.isclose(result, expected_result)

    @pytest.mark.parametrize(('x, y'), [(1.0, 1.0j), (1.0j, 1.0),
            (1.0, 1.0), (1.0j, 1.0j)])
    def test_dot_product(self, x, y):
        cbuild = CodeBuilder()
        cbuild.add_and_get_ids(
            AssignExpression(id='assign_1', assignee='x',
                             expression=var('<builtin>dot_product')(x, y)),
            YieldState(id='return', time=0, time_id='final',
                        expression=var('x'), component_id='<state>',
                        depends_on=['assign_1']))
        cbuild.commit()
        code = TimeIntegratorCode(initialization_dep_on=[],
                                  instructions=cbuild.instructions,
                                  step_dep_on=['return'],
                                  step_before_fail=False)
        result = self.execute_and_return_single_result(code)
        assert result == np.vdot(x, y)


class TestBuiltinsWithNumpyInterpreter(BuiltinsTestBase):

    def execute_and_return_single_result(self, code):
        from leap.vm.exec_numpy import NumpyInterpreter
        interpreter = NumpyInterpreter(code, function_map={})
        interpreter.set_up(t_start=0, dt_start=0, state={})
        interpreter.initialize()
        events = [event for event in interpreter.run(t_end=0)]
        assert len(events) == 2
        assert isinstance(events[0], interpreter.StateComputed)
        assert isinstance(events[1], interpreter.StepCompleted)
        return events[0].state_component


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
