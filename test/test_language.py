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
from leap.vm.language import (CodeBuilder, SimpleCodeBuilder,
                              TimeIntegratorCode)
from pymbolic import var
from pymbolic.primitives import Comparison

from leap.vm.exec_numpy import NumpyInterpreter  # noqa
from leap.vm.codegen import PythonCodeGenerator  # noqa


def test_SimpleCodeBuilder_yield(execute_and_return_single_result):
    cb = CodeBuilder()
    with SimpleCodeBuilder(cb) as builder:
        yield_ = builder.yield_state(1, 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_init_and_step(
            [], yield_, cb.instructions, True)
    result = execute_and_return_single_result(code)
    assert result == 1


def test_SimpleCodeBuilder_assign(execute_and_return_single_result):
    cb = CodeBuilder()
    with SimpleCodeBuilder(cb) as builder:
        builder.assign(var('x'), 1)
        yield_ = builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_init_and_step(
            [], yield_, cb.instructions, True)
    result = execute_and_return_single_result(code)
    assert result == 1


def test_SimpleCodeBuilder_condition(execute_and_return_single_result):
    cb = CodeBuilder()
    with SimpleCodeBuilder(cb) as builder:
        builder.assign(var('x'), 1)
        with builder.condition(Comparison(var('x'), '==', 1)):
            builder.assign(var('x'), 2)
        yield_ = builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_init_and_step(
            [], yield_, cb.instructions, True)
    result = execute_and_return_single_result(code)
    assert result == 2


def test_SimpleCodeBuilder_nested_condition(execute_and_return_single_result):
    cb = CodeBuilder()
    with SimpleCodeBuilder(cb) as builder:
        builder.assign(var('x'), 1)
        with builder.condition(Comparison(var('x'), '==', 1)):
            builder.assign(var('x'), 2)
            with builder.condition(Comparison(var('x'), '==', 2)):
                builder.assign(var('x'), 3)
            yield_ = builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_init_and_step(
            [], yield_, cb.instructions, True)
    result = execute_and_return_single_result(code)
    assert result == 3


def test_SimpleCodeBuilder_dependencies(execute_and_return_single_result):
    cb = CodeBuilder()
    with SimpleCodeBuilder(cb) as builder:
        dependency = builder.assign(var('x'), 1)
    with SimpleCodeBuilder(cb, dependency) as builder:
        yield_ = builder.yield_state(var('x'), 'x', 0, 'final')
    code = TimeIntegratorCode.create_with_init_and_step(
            [], yield_, cb.instructions, True)
    result = execute_and_return_single_result(code)
    assert result == 1


def test_collapse_constants():
    from pymbolic import var
    f = var("f")
    y = var("y")
    t = var("t")
    dt = var("dt")
    expr = y - f(t + dt, y)
    from leap.vm.expression import collapse_constants
    from pymbolic import var

    def new_var_func():
        return var("var")

    def assign_func(variable, expr):
        assert variable == var("var")
        assert expr == t + dt

    collapse_constants(expr, [y], assign_func, new_var_func)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
