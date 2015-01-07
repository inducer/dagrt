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

import sys
from leap.vm.language import (NewCodeBuilder, TimeIntegratorCode)
from pymbolic import var

from leap.vm.exec_numpy import NumpyInterpreter  # noqa
from leap.vm.codegen import PythonCodeGenerator  # noqa

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)

from utils import execute_and_return_single_result


def test_NewCodeBuilder_yield(python_method_impl):
    with NewCodeBuilder() as builder:
        builder.yield_state(1, 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_steady_state(
        builder.state_dependencies, builder.instructions)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 1


def test_NewCodeBuilder_assign(python_method_impl):
    with NewCodeBuilder() as builder:
        builder(var('x'), 1)
        builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_steady_state(
        builder.state_dependencies, builder.instructions)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 1


def test_NewCodeBuilder_condition(python_method_impl):
    with NewCodeBuilder() as builder:
        builder(var('x'), 1)
        with builder.if_(var('x'), '==', 1):
            builder(var('x'), 2)
        builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_steady_state(
        builder.state_dependencies, builder.instructions)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 2


def test_NewCodeBuilder_condition_with_else(python_method_impl):
    with NewCodeBuilder() as builder:
        builder(var('x'), 1)
        with builder.if_(var('x'), '!=', 1):
            builder(var('x'), 2)
        with builder.else_():
            builder(var('x'), 3)
        builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_steady_state(
        builder.state_dependencies, builder.instructions)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 3


def test_NewCodeBuilder_nested_condition(python_method_impl):
    with NewCodeBuilder() as builder:
        builder(var('x'), 1)
        with builder.if_(var('x'), '==', 1):
            builder(var('x'), 2)
            with builder.if_(var('x'), '==', 2):
                builder(var('x'), 3)
            builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_steady_state(
        builder.state_dependencies, builder.instructions)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 3


def test_NewCodeBuilder_nested_condition_with_else(python_method_impl):
    with NewCodeBuilder() as builder:
        builder(var('x'), 1)
        with builder.if_(var('x'), '==', 1):
            builder(var('x'), 2)
            with builder.if_(var('x'), '!=', 2):
                builder(var('x'), 3)
            with builder.else_():
                builder(var('x'), 4)
            builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_steady_state(
        builder.state_dependencies, builder.instructions)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 4


def test_collapse_constants():
    from pymbolic import var
    f = var("f")
    y = var("y")
    t = var("t")
    dt = var("dt")
    expr = y - f(t + dt, y)
    from leap.vm.expression import collapse_constants

    def new_var_func():
        return var("var")

    def assign_func(variable, expr):
        assert variable == var("var")
        assert expr == t + dt

    collapse_constants(expr, [y], assign_func, new_var_func)


def test_unify():
    from pymbolic import var
    f = var("f")
    y = var("y")
    h = var("h")
    t = var("t")
    hh = var("hh")
    tt = var("tt")
    lhs = y - h * f(t, y)
    rhs = - hh * f(tt, y) + y

    from leap.vm.expression import unify
    subst = unify(lhs, rhs, ["t", "h"])
    assert len(subst) == 2
    assert subst["h"] == hh
    assert subst["t"] == tt


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
