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

from leap.vm.language import AssignExpression, If, YieldState, FailStep, Raise
from leap.vm.language import CodeBuilder, TimeIntegratorCode
from leap.vm.codegen import PythonCodeGenerator
from pymbolic import var


def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id='return', time=0, time_id='final',
                    expression=0, component_id='<state>',
        depends_on=[]))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name='Method')
    Method = codegen.get_class(code)
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    method.initialize()
    hist = [s for s in method.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], method.StateComputed)
    assert hist[0].state_component == 0
    assert isinstance(hist[1], method.StepCompleted)


def test_basic_conditional_codegen():
    """Test whether the code generator generates branches properly."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='then_branch', assignee='<state>y', expression=1),
        AssignExpression(id='else_branch', assignee='<state>y', expression=0),
        If(id='branch', condition=True, then_depends_on=['then_branch'],
            else_depends_on=['else_branch']),
        YieldState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
        depends_on=['branch']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name='Method')
    Method = codegen.get_class(code)
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={'y': 6})
    method.initialize()
    hist = [s for s in method.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], method.StateComputed)
    assert hist[0].state_component == 1
    assert isinstance(hist[1], method.StepCompleted)


def test_basic_assign_rhs_codegen():
    """Test whether the code generator generates RHS evaluation code
    properly."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_rhs1',
                         assignee='<state>y',
                         expression=var('y')(t=var('<t>')),
                         depends_on=[]),
        AssignExpression(id='assign_rhs2',
                         assignee='<state>y',
                         expression=var('yy')(t=var('<t>'), y=var('<state>y')),
                         depends_on=['assign_rhs1']),
        YieldState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
            depends_on=['assign_rhs2'])
        )
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name='Method')
    Method = codegen.get_class(code)

    def y(t):
        return 6

    def yy(t, y):
        return y + 6

    method = Method({'y': y, 'yy': yy})
    method.set_up(t_start=0, dt_start=0, context={'y': 0})
    method.initialize()
    hist = [s for s in method.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], method.StateComputed)
    assert hist[0].state_component == 12
    assert isinstance(hist[1], method.StepCompleted)


def test_basic_raise_codegen():
    """Test code generation of the Raise instruction."""
    cbuild = CodeBuilder()
    from leap.method import TimeStepUnderflow
    cbuild.add_and_get_ids(Raise(TimeStepUnderflow, "underflow", id="raise"))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=["raise"],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    method.initialize()
    try:
        for result in method.run(t_end=0):
            assert False
    except method.TimeStepUnderflow:
        pass
    except:
        assert False


def test_basic_fail_step_codegen():
    """Test code generation of the Raise instruction."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(FailStep(id="fail"))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=["fail"],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    method.initialize()
    assert isinstance(next(method.run(t_end=0)), method.StepFailed)


def test_local_name_distinctness():
    """Test whether the code generator gives locals distinct names."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_y^', assignee='y^', expression=1),
        AssignExpression(id='assign_y*', assignee='y*', expression=0),
        YieldState(id='return', time=0, time_id='final',
            expression=var('y^') + var('y*'),
            component_id='y', depends_on=['assign_y^', 'assign_y*']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name='Method')
    Method = codegen.get_class(code)
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    method.initialize()
    hist = list(method.run(t_end=0))
    assert len(hist) == 2
    assert isinstance(hist[0], method.StateComputed)
    assert hist[0].state_component == 1


def test_global_name_distinctness():
    """Test whether the code generator gives globals distinct names."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign_y^', assignee='<p>y^', expression=1),
        AssignExpression(id='assign_y*', assignee='<p>y*', expression=0),
        YieldState(id='return', time=0, time_id='final',
            expression=var('<p>y^') + var('<p>y*'),
            component_id='y', depends_on=['assign_y^', 'assign_y*']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name='Method')
    Method = codegen.get_class(code)
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    method.initialize()
    hist = list(method.run(t_end=0))
    assert len(hist) == 2
    assert isinstance(hist[0], method.StateComputed)
    assert hist[0].state_component == 1


def test_function_name_distinctness():
    """Test whether the code generator gives functions distinct names."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id='return', time=0, time_id='final',
            expression=var('<func>y^')() + var('<func>y*')(),
            component_id='y'))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'],
            step_before_fail=False)
    codegen = PythonCodeGenerator(class_name='Method')
    Method = codegen.get_class(code)
    method = Method({'<func>y^': lambda: 0,
                     '<func>y*': lambda: 1})
    method.set_up(t_start=0, dt_start=0, context={})
    method.initialize()
    hist = list(method.run(t_end=0))
    assert len(hist) == 2
    assert isinstance(hist[0], method.StateComputed)
    assert hist[0].state_component == 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
