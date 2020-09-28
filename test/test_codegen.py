#! /usr/bin/env python

import pytest
import sys

from dagrt.language import Assign, YieldState
from dagrt.codegen import CodeGenerationError
from dagrt.codegen.analysis import verify_code

from pymbolic import var

from utils import (
        RawCodeBuilder,
        create_DAGCode_with_init_and_main_phases,
        create_DAGCode_with_steady_phase)


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
        Assign(
            id="assign",
            assignee="<state>y",
            assignee_subscript=(),
            expression=1,
            depends_on=["assign2"]),
        Assign(id="assign2",
            assignee="<state>y",
            assignee_subscript=(),
            expression=1,
            depends_on=["assign"]),
        YieldState(id="return", time=0, time_id="final",
            expression=var("<state>y"), component_id="<state>",
        depends_on=["assign"]))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    with pytest.raises(CodeGenerationError):
        verify_code(code)


def test_missing_dependency_detection():
    """Check that the code generator detects that there is a missing
    dependency."""
    statements = {
        Assign(id="assign", assignee="<state>y", assignee_subscript=(),
            expression=1, depends_on=["assign2"]),
        YieldState(id="return", time=0, time_id="final",
            expression=var("<state>y"), component_id="<state>",
            depends_on=["assign"])
        }
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=statements)
    with pytest.raises(CodeGenerationError):
        verify_code(code)


def test_missing_state_detection():
    """Check that the code generator detects there is a missing state."""
    from dagrt.language import CodeBuilder

    with CodeBuilder(name="state_1") as cb:
        cb.switch_phase("state_2")

    code = create_DAGCode_with_steady_phase(statements=cb.statements)
    with pytest.raises(CodeGenerationError):
        verify_code(code)


def test_cond_detection():
    """Check that the code generator detects a redefinition of a <cond> variable."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        Assign(
            id="assign1",
            assignee="<cond>c",
            assignee_subscript=(),
            expression=1,
            depends_on=[]),
        Assign(
            id="assign2",
            assignee="<cond>c",
            assignee_subscript=(),
            expression=2,
            depends_on=["assign1"]),
        YieldState(id="return",
            time=0, time_id="final",
            expression=1,
            component_id="<state>",
            depends_on=["assign2"]))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    with pytest.raises(CodeGenerationError):
        verify_code(code)


def test_python_line_wrapping():
    """Check that the line wrapper breaks a line up correctly."""
    from dagrt.codegen.python import wrap_line
    line = "x += str('' + x + y + zzzzzzzzz)"
    result = wrap_line(line, level=1, width=14, indentation="    ")
    assert result == ["x +=     \\", "    str(''\\", "    + x +\\",
                      "    y +  \\", "    zzzzzzzzz)"]


def test_line_wrapping_line_with_string():
    """Check that the line wrapper doesn't break up strings."""
    from dagrt.codegen.fortran import wrap_line
    line = "write(*,*) 'failed to allocate dagrt_state%dagrt_refcnt_p_last_rhs_y'"
    result = wrap_line(line, width=60)
    assert result == \
        ["write(*,*)                                                 &",
         "    'failed to allocate dagrt_state%dagrt_refcnt_p_last_rhs_y'"]


def test_KeyToUniqueNameMap():
    from dagrt.codegen.utils import KeyToUniqueNameMap

    map_prefilled = KeyToUniqueNameMap(start={"a": "b"})
    assert map_prefilled.get_or_make_name_for_key("a") == "b"
    assert map_prefilled.get_or_make_name_for_key("b") != "b"

    map_with_prefix = KeyToUniqueNameMap(forced_prefix="prefix")
    assert map_with_prefix.get_or_make_name_for_key("a") == "prefixa"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
