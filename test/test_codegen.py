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

from leap.vm.language import AssignExpression, YieldState
from leap.vm.language import CodeBuilder, TimeIntegratorCode
from leap.vm.codegen import PythonCodeGenerator, CodeGenerationError
from pymbolic import var
from leap.vm.codegen.ir import BasicBlock, SymbolTable, Function
from leap.vm.codegen.structured_ir import SingleNode, BlockNode, IfThenNode, \
    IfThenElseNode, UnstructuredIntervalNode
from leap.vm.codegen.ir2structured_ir import StructuralExtractor
from pytools import one


def test_circular_dependency_detection():
    """Check that the code generator detects that there is a circular
    dependency."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(id='assign', assignee='<state>y', expression=1,
                         depends_on=['assign2']),
        AssignExpression(id='assign2', assignee='<state>y', expression=1,
                         depends_on=['assign']),
        YieldState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
        depends_on=['assign']))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'])
    codegen = PythonCodeGenerator(class_name='Method')
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
        YieldState(id='return', time=0, time_id='final',
            expression=var('<state>y'), component_id='<state>',
            depends_on=['assign'])
        ])
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=instructions, step_dep_on=['return'])
    codegen = PythonCodeGenerator(class_name='Method')
    try:
        codegen(code)
    except CodeGenerationError:
        pass
    else:
        assert False


def test_missing_state_detection():
    """Check that the code generator detects there is a missing state."""
    from leap.vm.language import NewCodeBuilder

    with NewCodeBuilder(label="state_1") as cb:
        cb.state_transition("state_2")

    code = TimeIntegratorCode.create_with_steady_state(
        dep_on=cb.state_dependencies, instructions=cb.instructions)

    from leap.vm.codegen.analysis import verify_code
    try:
        verify_code(code)
    except CodeGenerationError:
        pass
    else:
        assert False


def test_basic_structural_extraction():
    """Check that structural extraction correctly detects a basic block."""
    sym_tab = SymbolTable()
    main = Function("f", sym_tab)
    block = BasicBlock(0, main)
    main.assign_entry_block(block)

    block.add_bogus_yield_state()
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
    main = Function("f", sym_tab)

    blocks = [BasicBlock(i, main) for i in range(0, 4)]
    for i, block in enumerate(blocks):
        if i == 3:
            block.add_bogus_yield_state()
        else:
            block.add_jump(blocks[i + 1])
    main.assign_entry_block(blocks[0])

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
    main = Function("f", sym_tab)
    blocks = [BasicBlock(i, main) for i in range(0, 3)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[2])
    blocks[2].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

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
    main = Function("f", sym_tab)
    blocks = [BasicBlock(i, main) for i in range(0, 4)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[3])
    blocks[2].add_jump(blocks[3])
    blocks[3].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

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
    main = Function("f", sym_tab)
    blocks = [BasicBlock(i, main) for i in range(0, 6)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[3])
    blocks[2].add_branch(None, blocks[3], blocks[4])
    blocks[3].add_jump(blocks[5])
    blocks[4].add_jump(blocks[5])
    blocks[5].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, UnstructuredIntervalNode)
    assert set(blocks) == set(node.basic_block for node in control_tree.nodes)


def test_unstructured_interval_structural_extraction_2():
    """Check that structural extraction correctly classifies a sequence with a
    basic block with a self loop as an unstructured interval.
    """
    sym_tab = SymbolTable()
    main = Function("f", sym_tab)
    blocks = [BasicBlock(i, main) for i in range(0, 2)]
    blocks[0].add_branch(None, blocks[0], blocks[1])
    blocks[1].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

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
    main = Function("f", sym_tab)

    blocks = [BasicBlock(i, main) for i in range(0, 6)]
    blocks[0].add_branch(None, blocks[1], blocks[3])
    blocks[1].add_branch(None, blocks[2], blocks[5])
    blocks[2].add_jump(blocks[5])
    blocks[3].add_branch(None, blocks[5], blocks[4])
    blocks[4].add_jump(blocks[5])
    blocks[5].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

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
    """Check that structural extraction correctly detects a complex
    control tree. This complex control tree is an If-Then-Else node
    containing two If-Then structures in blocks.

            0
           / \
          1   4
         /|  /|
         2|  5|
         \|  \|
          3   6

    """
    sym_tab = SymbolTable()
    main = Function("f", sym_tab)

    blocks = [BasicBlock(i, main) for i in range(0, 7)]
    blocks[0].add_branch(None, blocks[1], blocks[4])
    blocks[1].add_branch(None, blocks[2], blocks[3])
    blocks[2].add_jump(blocks[3])
    blocks[3].add_bogus_yield_state()
    blocks[4].add_branch(None, blocks[5], blocks[6])
    blocks[5].add_jump(blocks[6])
    blocks[6].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, IfThenElseNode)
    node_a, node_b = control_tree.then_node, control_tree.else_node
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
    main = Function("f", sym_tab)

    blocks = [BasicBlock(i, main) for i in range(0, 6)]
    blocks[0].add_branch(None, blocks[1], blocks[2])
    blocks[1].add_jump(blocks[3])
    blocks[2].add_jump(blocks[3])
    blocks[3].add_branch(None, blocks[4], blocks[5])
    blocks[4].add_jump(blocks[5])
    blocks[5].add_bogus_yield_state()
    main.assign_entry_block(blocks[0])

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


def test_cycle_detection():
    sym_tab = SymbolTable()
    main = Function("f", sym_tab)

    blocks = [BasicBlock(i, main) for i in range(3)]
    blocks[0].add_jump(blocks[1])
    blocks[1].add_jump(blocks[2])
    blocks[2].add_jump(blocks[1])
    main.assign_entry_block(blocks[0])

    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)
    assert isinstance(control_tree, UnstructuredIntervalNode)


def test_if_then_with_dangling_then():
    """
       0
       | \
       |  1
       |  |\
       |  | 2
       | /
       |/
       3
    """
    sym_tab = SymbolTable()
    main = Function("f", sym_tab)

    blocks = [BasicBlock(i, main) for i in range(4)]
    blocks[0].add_branch(None, blocks[1], blocks[3])
    blocks[1].add_branch(None, blocks[2], blocks[3])

    main.assign_entry_block(blocks[0])
    structural_extractor = StructuralExtractor()
    control_tree = structural_extractor(main)

    assert isinstance(control_tree, BlockNode)
    assert control_tree.blocks() == frozenset([0, 1, 2, 3])

    assert isinstance(control_tree.node_list[0], IfThenNode)
    assert control_tree.node_list[0].if_node.blocks() == frozenset([0])

    inner_if = control_tree.node_list[0].then_node
    assert isinstance(inner_if, IfThenNode)
    assert inner_if.if_node.blocks() == frozenset([1])
    assert inner_if.then_node.blocks() == frozenset([2])

    assert isinstance(control_tree.node_list[1], SingleNode)
    assert control_tree.node_list[1].blocks() == frozenset([3])


def test_python_line_wrapping():
    """Check that the line wrapper breaks a line up correctly."""
    from leap.vm.codegen.python import wrap_line
    line = "x += str('' + x + y + zzzzzzzzz)"
    result = wrap_line(line, level=1, width=14, indentation='    ')
    assert result == ['x +=     \\', "    str(''\\", '    + x +\\',
                      '    y +  \\', '    zzzzzzzzz)']


def test_line_wrapping_line_with_string():
    """Check that the line wrapper doesn't break up strings."""
    from leap.vm.codegen.fortran import wrap_line
    line = "write(*,*) 'failed to allocate leap_state%leap_refcnt_p_last_rhs_y'"
    result = wrap_line(line, width=60)
    assert result == \
        ["write(*,*)                                                 &",
         "    'failed to allocate leap_state%leap_refcnt_p_last_rhs_y'"]


def test_KeyToUniqueNameMap():
    from leap.vm.codegen.utils import KeyToUniqueNameMap

    map_prefilled = KeyToUniqueNameMap(start={'a': 'b'})
    assert map_prefilled.get_or_make_name_for_key('a') == 'b'
    assert map_prefilled.get_or_make_name_for_key('b') != 'b'

    map_with_prefix = KeyToUniqueNameMap(forced_prefix='prefix')
    assert map_with_prefix.get_or_make_name_for_key('a') == 'prefixa'


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
