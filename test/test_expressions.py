#! /usr/bin/env python

from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2014, 2015 Matt Wala"

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


def test_collapse_constants():
    from pymbolic import var
    f = var("f")
    y = var("y")
    t = var("t")
    dt = var("dt")
    expr = y - f(t + dt, y)
    from dagrt.vm.expression import collapse_constants

    def new_var_func():
        return var("var")

    def assign_func(variable, expr):
        assert variable == var("var")
        assert expr == t + dt

    collapse_constants(expr, [y], assign_func, new_var_func)


def test_match():
    from pymbolic import var
    f = var("f")
    y = var("y")
    h = var("h")
    t = var("t")
    hh = var("hh")
    tt = var("tt")
    lhs = y - h * f(t, y)
    rhs = - hh * f(tt, y) + y

    from dagrt.vm.expression import match
    subst = match(lhs, rhs, ["t", "h"])
    assert len(subst) == 2
    assert subst["h"] == hh
    assert subst["t"] == tt


def test_parse():
    from pymbolic import var
    from dagrt.vm.expression import parse
    assert parse("1 + `<dt>`") == 1 + var("<dt>")


def test_get_variables():
    from pymbolic import var
    f = var('f')
    x = var('x')
    from dagrt.vm.utils import get_variables
    assert get_variables(f(x)) == frozenset(['x'])
    assert get_variables(f(t=x)) == frozenset(['x'])


def test_get_variables_with_function_symbols():
    from pymbolic import var
    f = var('f')
    x = var('x')
    from dagrt.vm.utils import get_variables
    assert get_variables(f(x), include_function_symbols=True) == \
        frozenset(['f', 'x'])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
