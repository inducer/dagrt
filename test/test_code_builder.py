#! /usr/bin/env python

import sys
from dagrt.language import (CodeBuilder, DAGCode)
from pymbolic import var

from dagrt.exec_numpy import NumpyInterpreter  # noqa
from dagrt.codegen import PythonCodeGenerator  # noqa

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)

from utils import (
        execute_and_return_single_result,
        create_DAGCode_with_steady_phase)


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


def test_CodeBuilder_yield(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder.yield_state(1, "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 1


def test_CodeBuilder_assign(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 1


def test_CodeBuilder_condition(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        with builder.if_(var("x"), "==", 1):
            builder(var("x"), 2)
        builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 2


def test_CodeBuilder_write_ordering(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder("y", "1")
        builder("x", "y")
        builder("y", "2")
        builder("z", "y")
        builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 1


def test_CodeBuilder_condition_with_else(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        with builder.if_(var("x"), "!=", 1):
            builder(var("x"), 2)
        with builder.else_():
            builder(var("x"), 3)
        builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 3


def test_CodeBuilder_condition_with_else_not_taken(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        with builder.if_(var("x"), "==", 1):
            builder(var("x"), 2)
        with builder.else_():
            builder(var("x"), 3)
        builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 2


def test_CodeBuilder_nested_condition(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        with builder.if_(var("x"), "==", 1):
            builder(var("x"), 2)
            with builder.if_(var("x"), "==", 2):
                builder(var("x"), 3)
            builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 3


def test_CodeBuilder_nested_condition_with_else(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        with builder.if_(var("x"), "==", 1):
            builder(var("x"), 2)
            with builder.if_(var("x"), "!=", 2):
                builder(var("x"), 3)
            with builder.else_():
                builder(var("x"), 4)
            builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 4


def test_CodeBuilder_nested_condition_with_else_not_taken(python_method_impl):
    with CodeBuilder("phase") as builder:
        builder(var("x"), 1)
        with builder.if_(var("x"), "==", 1):
            builder(var("x"), 2)
            with builder.if_(var("x"), "==", 2):
                builder(var("x"), 3)
            with builder.else_():
                builder(var("x"), 4)
            builder.yield_state(var("x"), "x", 0, "final")
    code = create_DAGCode_with_steady_phase(builder.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 3


def test_CodeBuilder_restart_step(python_method_impl):
    with CodeBuilder("init") as builder_init:
        builder_init("<p>x", "0")

    with CodeBuilder("state1") as builder1:
        builder1("<p>x", "<p>x + 1")
        with builder1.if_("<p>x == 1"):
            builder1.restart_step()

    with CodeBuilder("state2") as builder2:
        builder2.yield_state(var("<p>x"), "x", 0, "final")

    phases = [
            builder_init.as_execution_phase(next_phase="state1"),
            builder1.as_execution_phase(next_phase="state2"),
            builder2.as_execution_phase(next_phase="state2")
    ]
    code = DAGCode.from_phases_list(phases, "init")

    result = execute_and_return_single_result(python_method_impl, code, max_steps=4)
    assert result == 2


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
