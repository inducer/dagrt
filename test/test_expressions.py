#! /usr/bin/env python

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


import pytest


def test_collapse_constants():
    from pymbolic import var
    f = var("f")
    y = var("y")
    t = var("t")
    dt = var("dt")
    expr = y - f(t + dt, y)
    from dagrt.expression import collapse_constants

    def new_var_func():
        return var("var")

    def assign_func(variable, expr):
        assert variable == var("var")
        assert expr == t + dt

    collapse_constants(expr, [y], assign_func, new_var_func)


def declare(*varnames):
    from pymbolic import var
    return [var(name) for name in varnames]


def test_match():
    f, y, h, t, yy, hh, tt = declare("f", "y", "h", "t", "yy", "hh", "tt")
    lhs = y - h * f(t, y)
    rhs = - hh * f(tt, yy) + yy

    from dagrt.expression import match
    subst = match(lhs, rhs, ["t", "h", "y"])
    assert len(subst) == 3
    assert subst["h"] == hh
    assert subst["t"] == tt
    assert subst["y"] == yy


def test_match_strings():
    from dagrt.expression import match
    from pymbolic import var
    subst = match("a+b*a", "a+b*a")
    assert len(subst) == 2
    assert subst["a"] == var("a")
    assert subst["b"] == var("b")


def test_match_functions():
    lhsvars = ["f", "u", "s", "c", "t"]
    f, u, s, c, t = declare(*lhsvars)
    ff, uu, ss, cc, tt = declare("ff", "uu", "ss", "cc", "tt")
    rhsvars = [ff, uu, ss, cc, tt]
    lhs = u - f(t=t, y=s+c*u)
    rhs = uu - ff(t=tt, y=ss+cc*uu)

    from dagrt.expression import match
    subst = match(lhs, rhs, lhsvars)
    assert len(subst) == len(lhsvars)
    for var, matchval in zip(lhsvars, rhsvars):
        assert subst[var] == matchval


def test_match_modulo_identity():
    a, b, c = declare("a", "b", "c")
    from dagrt.expression import match

    subst = match(c*a + b*a, c*a + a, ["b"])
    assert subst["b"] == 1

    subst = match((c+a) * (b+a), (c+a) * a, ["b"])
    assert subst["b"] == 0


def test_match_with_pre_match():
    a, b, c, d = declare("a", "b", "c", "d")
    from dagrt.expression import match
    subst = match(a + b, c + d, ["a", "b"], pre_match={"a": "c"})

    assert subst["a"] == c
    assert subst["b"] == d


def test_match_with_pre_match_invalid_arg():
    a, b, c, d = declare("a", "b", "c", "d")
    from dagrt.expression import match
    with pytest.raises(ValueError):
        match(a + b, c + d, ["a"], pre_match={"b": "c"})


def test_get_variables():
    from pymbolic import var
    f = var("f")
    x = var("x")
    from dagrt.utils import get_variables
    assert get_variables(f(x)) == frozenset(["x"])
    assert get_variables(f(t=x)) == frozenset(["x"])


def test_get_variables_with_function_symbols():
    from pymbolic import var
    f = var("f")
    x = var("x")
    from dagrt.utils import get_variables
    assert get_variables(f(x), include_function_symbols=True) == \
        frozenset(["f", "x"])


def test_substitute():
    f, a = declare("f", "a")

    from dagrt.expression import substitute
    assert substitute("f(<state>y)", {"<state>y": a}) == f(a)


# {{{ parser

def test_parser():
    from pymbolic import var
    from dagrt.expression import parse
    parse("(2*a[1]*b[1]+2*a[0]*b[0])*(hankel_1(-1,sqrt(a[1]**2+a[0]**2)*k) "
            "-hankel_1(1,sqrt(a[1]**2+a[0]**2)*k))*k /(4*sqrt(a[1]**2+a[0]**2)) "
            "+hankel_1(0,sqrt(a[1]**2+a[0]**2)*k)")
    print(repr(parse("d4knl0")))
    print(repr(parse("0.")))
    print(repr(parse("0.e1")))
    assert parse("0.e1") == 0
    assert parse("1e-12") == 1e-12
    print(repr(parse("a >= 1")))
    print(repr(parse("a <= 1")))

    print(repr(parse("g[i,k]+2.0*h[i,k]")))
    print(repr(parse("g[i,k]+(+2.0)*h[i,k]")))
    print(repr(parse("a - b - c")))
    print(repr(parse("-a - -b - -c")))
    print(repr(parse("- - - a - - - - b - - - - - c")))

    print(repr(parse("~(a ^ b)")))
    print(repr(parse("(a | b) | ~(~a & ~b)")))

    print(repr(parse("3 << 1")))
    print(repr(parse("1 >> 3")))

    print(parse("3::1"))

    import pymbolic.primitives as prim
    assert parse("e1") == prim.Variable("e1")
    assert parse("d1") == prim.Variable("d1")

    from pymbolic import variables
    f, x, y, z = variables("f x y z")
    assert parse("f((x,y),z)") == f((x, y), z)
    assert parse("f((x,),z)") == f((x,), z)
    assert parse("f(x,(y,z),z)") == f(x, (y, z), z)

    assert parse("f(x,(y,z),z, name=15)") == f(x, (y, z), z, name=15)
    assert parse("f(x,(y,z),z, name=15, name2=17)") == f(
            x, (y, z), z, name=15, name2=17)

    assert parse("<func>yoink") == var("<func>yoink")
    assert parse("-<  func  >  yoink") == -var("<func>yoink")
    print(repr(parse("<func>yoink < <p>x")))
    print(repr(parse("<func>yoink < - <p>x")))

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
