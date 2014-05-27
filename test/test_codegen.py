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
import numpy as np

from leap.vm.language import AssignExpression, AssignRHS, If, ReturnState
from leap.vm.language import CodeBuilder, TimeIntegratorCode
from leap.vm.exec_numpy import StateComputed, StepCompleted
from leap.vm.codegen import PythonCodeGenerator
from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper
from pymbolic import var


def exec_in_new_namespace(code):
    """Executes the given code with empty locals and returns the changes made
    to the locals namespace."""
    namespace = {}
    exec(code, globals(), namespace)
    return namespace


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
    codegen = PythonCodeGenerator(method_name='Method')
    output = codegen(code)
    state = exec_in_new_namespace(output)
    m = state['Method']({})
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
        AssignExpression(id='then_branch', assignee='<state>y', expression=1),
        AssignExpression(id='else_branch', assignee='<state>y', expression=0),
        If(id='branch', condition=True, then_depends_on=['then_branch'],
            else_depends_on=['else_branch']),
        ReturnState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
        depends_on=['branch']))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator(method_name='Method')
    output = codegen(code)
    state = exec_in_new_namespace(output)
    m = state['Method']({})
    m.set_up(t_start=0, dt_start=0, state={'y': 6})
    m.initialize()
    hist = [s for s in m.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], StateComputed)
    assert hist[0].state_component == 1
    assert isinstance(hist[1], StepCompleted)


def test_basic_assign_rhs_codegen():
    """Test whether the code generator generates RHS code properly."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignRHS(id='assign_rhs1', assignees=('<state>y',),
                  t=var('<t>'), component_id='y', depends_on=[],
                  rhs_arguments=((),)),
        AssignRHS(id='assign_rhs2', assignees=('<state>y',),
                  t=var('<t>'), component_id='yy', depends_on=['assign_rhs1'],
                  rhs_arguments=((('y', var('<state>y')),),)),
        ReturnState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
            depends_on=['assign_rhs2'])
        )
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator(method_name='Method')
    output = codegen(code)
    state = exec_in_new_namespace(output)

    def y(t):
        return 6

    def yy(t, y):
        return y + 6

    m = state['Method']({'y': y, 'yy': yy})
    m.set_up(t_start=0, dt_start=0, state={'y': 0})
    m.initialize()
    hist = [s for s in m.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], StateComputed)
    assert hist[0].state_component == 12
    assert isinstance(hist[1], StepCompleted)


def test_complex_dependency_codegen():
    """Test whether the code generator handles DAGs with complex layers of
    dependencies. In particular, check for correct handling of dependencies
    inside conditional code blocks."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='incr', assignee='<state>y', expression=
                         var('<state>y') + 1, depends_on=[]),
        If(id='branch1', condition=True, then_depends_on=['incr'],
           else_depends_on=[], depends_on=[]),
        If(id='branch2', condition=True, then_depends_on=['incr'],
           else_depends_on=[], depends_on=[]),
        ReturnState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
            depends_on=['branch1', 'branch2']))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator(method_name='Method')
    output = codegen(code)
    state = exec_in_new_namespace(output)
    m = state['Method']({})
    m.set_up(t_start=0, dt_start=0, state={'y': 0})
    m.initialize()
    hist = [s for s in m.run(t_end=0)]
    assert len(hist) == 2
    assert isinstance(hist[0], StateComputed)
    assert hist[0].state_component == 1
    assert isinstance(hist[1], StepCompleted)


@pytest.mark.parametrize(("stepper", "expected_order"), [
    (ODE23TimeStepper(use_high_order=False), 2),
    (ODE23TimeStepper(use_high_order=True), 3),
    (ODE45TimeStepper(use_high_order=False), 4),
    (ODE45TimeStepper(use_high_order=True), 5),
    ])
def test_rk_codegen(stepper, expected_order):
    """Test whether Runge-Kutta timestepper code generation works."""

    component_id = 'y'

    code = stepper(component_id)

    #from leap.vm.language import show_dependency_graph
    #show_dependency_graph(code)

    codegen = PythonCodeGenerator(method_name='RKMethod')
    output = codegen(code)
    state = exec_in_new_namespace(output)

    def rhs(t, y):
        u, v = y
        return np.array([v, -u/t**2], dtype=np.float64)

    def soln(t):
        inner = np.sqrt(3)/2*np.log(t)
        return np.sqrt(t)*(
                5*np.sqrt(3)/3*np.sin(inner)
                + np.cos(inner)
                )

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(4, 7):
        dt = 2**(-n)
        t = 1
        y = np.array([1, 3], dtype=np.float64)
        final_t = 10

        method = state['RKMethod']({component_id: rhs})
        method.set_up(t_start=t, dt_start=dt, state={component_id: y})
        method.initialize()

        times = []
        values = []

        for event in method.run(t_end=final_t):
            if isinstance(event, StateComputed):
                assert event.component_id == component_id
                values.append(event.state_component[0])
                times.append(event.t)

        assert abs(times[-1] - final_t) < 1e-10
        times = np.array(times)
        error = abs(values[-1]-soln(final_t))
        eocrec.add_data_point(dt, error)

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    print(eocrec.pretty_print())
    assert orderest > expected_order * 0.95


def test_multirate_codegen():
    """Test whether code generation works for multirate methods."""

    from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
    from leap.method.ab.multirate.methods import methods
    from pytools import DictionaryWithDefault

    order = DictionaryWithDefault(lambda x: 4)
    stepper = TwoRateAdamsBashforthTimeStepper(methods['F'], order, 4)

    code = stepper()
    codegen = PythonCodeGenerator(method_name='MRABMethod')
    output = codegen(code)
    namespace = exec_in_new_namespace(output)

    from ode_systems import Basic
    import numpy

    ode = Basic()

    # Requires a coupled component.
    def make_coupled(f2f, f2s, s2f, s2s):
        def coupled(t, y):
            args = (t, y[0] + y[1], y[2] + y[3])
            return numpy.array((f2f(*args), f2s(*args), s2f(*args),
                s2s(*args)),)
        return coupled

    rhs_map = {'<func>f2f': ode.f2f_rhs,
        '<func>s2f': ode.s2f_rhs, '<func>f2s': ode.f2s_rhs,
        '<func>s2s': ode.s2s_rhs, '<func>coupled': make_coupled(
            ode.f2f_rhs, ode.s2f_rhs, ode.f2s_rhs, ode.s2s_rhs)}

    from pytools.convergence import EOCRecorder

    eocrec = EOCRecorder()

    t = ode.t_start
    y = ode.initial_values
    final_t = ode.t_end
    for n in range(5, 8):
        dt = 2 ** -n

        method = namespace['MRABMethod'](rhs_map)
        method.set_up(t_start=t, dt_start=dt, state={'fast': y[0],
                                                     'slow': y[1]})
        method.initialize()

        times = []
        values = []
        for event in method.run(t_end=final_t):
            if isinstance(event, StateComputed):
                values.append(event.state_component)
                times.append(event.t)

        assert abs(times[-1] - final_t) < 1e-10

        tt = times[-1]
        yy = values[-1]

        error = abs(yy[0] - ode.soln_0(tt))

        eocrec.add_data_point(dt, error)

    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]

    assert orderest > 4 * 0.70


def test_circular_dependency_detection():
    """Check that the code generator detects that there is a circular
    dependency."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign', assignee='<state>y', expression=1,
                         depends_on=['assign2']),
        AssignExpression(id='assign2', assignee='<state>y', expression=1,
                         depends_on=['assign']),
        ReturnState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
        depends_on=['assign']))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator(method_name='Method')
    try:
        codegen(code)
    except PythonCodeGenerator.CodeGenerationError:
        pass
    else:
        assert False


def test_missing_dependency_detection():
    """Check that the code generator detects that there is a missing
    dependency."""
    instructions = set([
        AssignExpression(id='assign', assignee='<state>y', expression=1,
                         depends_on=['assign2']),
        ReturnState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
            depends_on=['assign'])
        ])
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = PythonCodeGenerator(method_name='Method')
    try:
        codegen(code)
    except PythonCodeGenerator.CodeGenerationError:
        pass
    else:
        assert False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
