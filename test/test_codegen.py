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
from leap.vm.codegen import PythonCodeGenerator, CodeGenerationError
from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper
from pymbolic import var
from leap.vm.codegen.ir import BasicBlock, SymbolTable, Function
from leap.vm.codegen.structured_ir import SingleNode, BlockNode, IfThenNode, \
    IfThenElseNode, UnstructuredIntervalNode
from leap.vm.codegen.ir2structured_ir import StructuralExtractor
from pytools import one


def exec_in_new_namespace(code):
    """Execute the given code with empty locals and returns the changes made to
    the locals namespace."""
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
        AssignExpression(id='incr', assignee='<state>y',
                         expression=var('<state>y') + 1, depends_on=[]),
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
    except CodeGenerationError:
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
    except CodeGenerationError:
        pass
    else:
        assert False


def test_basic_structural_extraction():
    """Check that structural extraction correctly detects a basic block."""
    block = BasicBlock(0, SymbolTable())
    block.add_return(None)
    main = Function(block)
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, SingleNode)
    assert len(control_tree.predecessors) == 0
    assert len(control_tree.successors) == 0
    assert len(control_tree.nodes) == 1 and block in control_tree.nodes
    assert control_tree.basic_block is block


def test_block_structural_extraction():
    """Check that structural extraction correctly detects a sequence of
    basic blocks.
    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 4)]
    for i, block in enumerate(blocks):
        if i == 3:
            block.add_return(None)
        else:
            block.add_jump(blocks[i + 1])
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, BlockNode)
    assert len(control_tree.nodes) == 4
    assert len(control_tree.node_list) == 4
    assert len(control_tree.predecessors) == 0
    assert len(control_tree.successors) == 0
    for i in range(0, 4):
        assert blocks[i] is control_tree.node_list[i].basic_block


def test_if_then_structural_extraction():
    """Check that structural extraction correctly detects an If-Then control
    structure.
    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 3)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[2])
    blocks[2].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, BlockNode)
    assert len(control_tree.node_list) == 2
    if_then_node = control_tree.node_list[0]
    assert isinstance(if_then_node, IfThenNode)
    assert blocks[0] is if_then_node.if_node.basic_block
    assert blocks[1] is if_then_node.then_node.basic_block
    assert blocks[2] is one(if_then_node.successors).basic_block


def test_if_then_else_structural_extraction():
    """Check that structural extraction correctly detects an If-Then-Else
    control structure.
    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 4)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[3])
    blocks[2].add_jump(blocks[3])
    blocks[3].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, BlockNode)
    assert len(control_tree.node_list) == 2
    if_then_else_node = control_tree.node_list[0]
    assert isinstance(if_then_else_node, IfThenElseNode)
    assert blocks[0] is if_then_else_node.if_node.basic_block
    assert blocks[1] is if_then_else_node.then_node.basic_block
    assert blocks[2] is if_then_else_node.else_node.basic_block
    assert blocks[3] is one(if_then_else_node.then_node.successors).basic_block
    assert blocks[3] is one(if_then_else_node.else_node.successors).basic_block


def test_unstructured_interval_structural_extraction():
    """Check that structural extraction correctly detects an unstructured
    interval.

    The interval cannot be broken down in terms of elementary control
    structures. It looks as follows:

        0
      /   \
      1   2
       \ / \
        3   4
         \ /
          5

    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 6)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[3])
    blocks[2].add_branch(None, blocks[3], blocks[4])
    blocks[3].add_jump(blocks[5])
    blocks[4].add_jump(blocks[5])
    blocks[5].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, UnstructuredIntervalNode)
    assert set(blocks) == set(node.basic_block for node in control_tree.nodes)


def test_unstructured_interval_structural_extraction_2():
    """Check that structural extraction correctly classifies a sequence with a
    basic block with a self loop as an unstructured interval.
    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 2)]
    blocks[0].add_branch(None, blocks[0], blocks[1])
    blocks[1].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, UnstructuredIntervalNode)
    assert len(control_tree.nodes) == 2


def test_complex_structural_extraction():
    """Check that structural extraction correctly detects a complex control
    tree. This complex control tree is a If-Then-Else control structure, with
    the Then and Else branches consisting of If-Then control structures.

            0
           /  \
          1    3
         / \  /|
         2 | / 4
          \|/ /
           5_/

    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 6)]
    blocks[0].add_branch(None, blocks[1], blocks[3])
    blocks[1].add_branch(None, blocks[2], blocks[5])
    blocks[2].add_jump(blocks[5])
    blocks[3].add_branch(None, blocks[5], blocks[4])
    blocks[4].add_jump(blocks[5])
    blocks[5].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, BlockNode)
    assert len(control_tree.node_list) == 2
    assert isinstance(control_tree.node_list[0], IfThenElseNode)
    assert isinstance(control_tree.node_list[1], SingleNode)
    if_then_else = control_tree.node_list[0]
    assert isinstance(if_then_else.then_node, IfThenNode)
    assert isinstance(if_then_else.else_node, IfThenNode)


def test_complex_structural_extraction_2():
    """Check that structural extraction correctly detects a complex control
    tree. This complex control tree is an unstructured interval containing two
    If-Then structures in blocks.

            0
           / \
          1   4
         /|  /|
         2|  5|
         \|  \|
          3   6

    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 7)]
    blocks[0].add_branch(None, blocks[1], blocks[4])
    blocks[1].add_branch(None, blocks[2], blocks[3])
    blocks[2].add_jump(blocks[3])
    blocks[3].add_return(None)
    blocks[4].add_branch(None, blocks[5], blocks[6])
    blocks[5].add_jump(blocks[6])
    blocks[6].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, UnstructuredIntervalNode)
    assert len(control_tree.nodes) == 3
    node_a, node_b, node_c = tuple(control_tree.nodes)
    if isinstance(node_a, SingleNode):
        node_a = node_c
    elif isinstance(node_b, SingleNode):
        node_b = node_c
    elif isinstance(node_c, SingleNode):
        pass
    else:
        assert False
    assert isinstance(node_a, BlockNode)
    assert isinstance(node_b, BlockNode)


def test_complex_structural_extraction_3():
    """Check that structural extraction correctly detects a complex control
    tree. This complex control tree is a block of If-Then-Else and If-Then
    control structures.

        0
       / \
       1 2
       \ /
        3
        |\
        | 4
        |/
        5

    """
    sym_tab = SymbolTable()
    blocks = [BasicBlock(i, sym_tab) for i in range(0, 6)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[3])
    blocks[2].add_jump(blocks[3])
    blocks[3].add_branch(None, blocks[4], blocks[5])
    blocks[4].add_jump(blocks[5])
    blocks[5].add_return(None)
    main = Function(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert len(control_tree.node_list) == 2
    assert isinstance(control_tree.node_list[0], IfThenNode)
    assert isinstance(control_tree.node_list[1], SingleNode)
    inner_if_node = control_tree.node_list[0].if_node
    assert isinstance(inner_if_node, BlockNode)
    assert len(inner_if_node.node_list) == 2
    assert isinstance(inner_if_node.node_list[0], IfThenElseNode)
    assert isinstance(inner_if_node.node_list[1], SingleNode)


def test_python_line_wrapping():
    """Check that the line wrapper breaks a line up correctly."""
    from leap.vm.codegen.python import wrap_line
    line = "x += str('' + x + y + zzzzzzzzz)"
    result = wrap_line(line, level=1, width=14, indentation='    ')
    assert result == ['x +=     \\', "    str(''\\", '    + x +\\',
                      '    y +  \\', '    zzzzzzzzz)']


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
