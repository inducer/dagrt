#! /usr/bin/env python

from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2015 Matt Wala"

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

from leap.vm.implicit import ScipySolverGenerator
from leap.vm.expression import parse
import numpy as np

import pytest


def test_ScipySolverGenerator_code_generation():
    pytest.importorskip("scipy")

    gen = ScipySolverGenerator(parse("z**2 - y"), "z")
    func = gen.get_compiled_solver()
    assert np.allclose(func(1, 2), np.sqrt(2))


def test_ScipySolverGenerator_code_generation_with_function_calls():
    pytest.importorskip("scipy")

    gen = ScipySolverGenerator(parse("`<func>f`(x, y=x)"), "x")
    func = gen.get_compiled_solver()

    def f(x, y):
        return (x + y) ** 2 - 3.0

    assert np.allclose(func(1.0, f), np.sqrt(3) / 2)


def test_ScipySolverGenerator_code_generation_with_builtins():
    pytest.importorskip("scipy")

    gen = ScipySolverGenerator(parse("`<builtin>dot_product`(x, x) - 1"), "x")
    func = gen.get_compiled_solver()
    assert np.allclose(func(-0.9), -1)


def test_ScipySolverGenerator_code_generation_with_vectors():
    pytest.importorskip("scipy")

    gen = ScipySolverGenerator(parse("x"), "x")
    func = gen.get_compiled_solver()
    assert np.allclose(func(np.array([-0.9, -0.1])), np.array([0.0, 0.0]))


def test_ScipySolverGenerator_callback():
    pytest.importorskip("scipy")

    gen = ScipySolverGenerator(parse("x + `<func>f`(x, b)"), "x")
    expr = parse("`<func>f`(y, b) + y")
    assert gen(expr, "y", 6) == parse("`<func>solver`(6, `<func>f`, b)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
