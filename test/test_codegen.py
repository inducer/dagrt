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

import pytest
import sys

from leap.vm.language import AssignExpression, If, ReturnState
from leap.vm.language import CodeBuilder, TimeIntegratorCode
from leap.vm.exec_numpy import StateComputed, StepCompleted
from leap.vm.codegen import PythonCodeGenerator
from pymbolic import var

def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        ReturnState(id='return', time=0, time_id='final',
                    expression=0, component_id='<state>',
        depends_on=[]))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator()
    output = codegen(code)
    exec(output)
    m = Method({})
    m.set_up(t_start=0, dt_start=0, state={})
    m.initialize()
    hist = [s for s in m.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], StateComputed)
    assert hist[0].state_component == 0
    assert isinstance(hist[1], StepCompleted)

def test_basic_conditional_codegen():
    """Test whether the code generator generates branches properly."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='then_branch', assignee='<state>', expression=1),
        AssignExpression(id='else_branch', assignee='<state>', expression=0),
        If(id='branch', condition=True, then_depends_on=['then_branch'],
            else_depends_on=['else_branch']),
        ReturnState(id='return', time=0, time_id='final',
            expression=var('<state>'), component_id='<state>',
        depends_on=['branch']))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator()
    output = codegen(code)
    print output
    exec(output)
    m = Method({})
    m.set_up(t_start=0, dt_start=0, state={'<state>': 6})
    m.initialize()
    hist = [s for s in m.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], StateComputed)
    assert hist[0].state_component == 1
    assert isinstance(hist[1], StepCompleted)
    
def test_basic_assign_rhs_codegen():
    pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])    
