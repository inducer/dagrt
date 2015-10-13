#! /usr/bin/env python
from __future__ import division, with_statement

import sys

from dagrt.language import AssignExpression, YieldState
from dagrt.language import TimeIntegratorCode
from dagrt.codegen import PythonCodeGenerator, CodeGenerationError

from pymbolic import var

from utils import RawCodeBuilder


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






def test_circular_dependency_detection():
    """Check that the code generator detects that there is a circular
    dependency."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        AssignExpression(
            id='assign',
            assignee='<state>y',
            assignee_subscript=(),
            expression=1,
            depends_on=['assign2']),
        AssignExpression(id='assign2', assignee='<state>y', assignee_subscript=(),
            expression=1, depends_on=['assign']),
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
        AssignExpression(id='assign', assignee='<state>y', assignee_subscript=(),
            expression=1, depends_on=['assign2']),
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
    from dagrt.language import CodeBuilder

    with CodeBuilder(label="state_1") as cb:
        cb.state_transition("state_2")

    code = TimeIntegratorCode.create_with_steady_state(
        dep_on=cb.state_dependencies, instructions=cb.instructions)

    from dagrt.codegen.analysis import verify_code
    try:
        verify_code(code)
    except CodeGenerationError:
        pass
    else:
        assert False


def test_python_line_wrapping():
    """Check that the line wrapper breaks a line up correctly."""
    from dagrt.codegen.python import wrap_line
    line = "x += str('' + x + y + zzzzzzzzz)"
    result = wrap_line(line, level=1, width=14, indentation='    ')
    assert result == ['x +=     \\', "    str(''\\", '    + x +\\',
                      '    y +  \\', '    zzzzzzzzz)']


def test_line_wrapping_line_with_string():
    """Check that the line wrapper doesn't break up strings."""
    from dagrt.codegen.fortran import wrap_line
    line = "write(*,*) 'failed to allocate leap_state%leap_refcnt_p_last_rhs_y'"
    result = wrap_line(line, width=60)
    assert result == \
        ["write(*,*)                                                 &",
         "    'failed to allocate leap_state%leap_refcnt_p_last_rhs_y'"]


def test_KeyToUniqueNameMap():
    from dagrt.codegen.utils import KeyToUniqueNameMap

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
