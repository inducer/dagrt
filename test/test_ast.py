#! /usr/bin/env python
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

import sys

from pymbolic import var

from pymbolic.primitives import LogicalNot

from dagrt.codegen.ast import (IfThen, IfThenElse, Block, InstructionWrapper,
                               create_ast_from_state, simplify_ast)
from dagrt.language import Instruction, DAGCode

import pytest


def test_create_ast():
    x = var("<cond>x")

    class ComparableNop(Instruction):
        """
        Works around the fact that __hash__ and comparison isn't defined for
        instructions, but is necessary for comparison purposes when embedded
        in an expression.
        """

        def _state(self):
            return (self.condition, self.id, self.depends_on)

        def __eq__(self, other):
            return (self.__class__ == other.__class__
                    and self._state() == other._state())

        def __ne__(self, other):
            return not self.__eq__(other)

        def __hash__(self):
            return hash(self._state())

    nop = ComparableNop(condition=x, id="nop", depends_on=())
    code = DAGCode.create_with_steady_state(["nop"], [nop])
    ast = create_ast_from_state(code, "main")
    assert ast == IfThen(x, InstructionWrapper(nop.copy(condition=True)))


_p = var("<cond>p")


@pytest.mark.parametrize("in_ast, out_ast", [
    (Block(Block(0)),
     0),

    (Block(IfThen(_p, 0), IfThen(_p, 1), IfThen(_p, 2)),
     IfThen(_p, Block(0, 1, 2))),

    (Block(IfThen(_p, 0), IfThen(LogicalNot(_p), 1)),
     IfThenElse(_p, 0, 1)),

    (Block(IfThen(_p, 0), IfThenElse(_p, 1, 2)),
     IfThenElse(_p, Block(0, 1), 2)),

    (Block(IfThenElse(_p, 0, 1), IfThen(LogicalNot(_p), 2)),
     IfThenElse(_p, 0, Block(1, 2))),

    (IfThenElse(_p, IfThen(LogicalNot(_p), 1), 2),
     IfThen(LogicalNot(_p), 2)),

    (Block(IfThenElse(_p, 1, 2), IfThenElse(_p, 3, 4)),
     IfThenElse(_p, Block(1, 3), Block(2, 4))),

    (Block(IfThenElse(LogicalNot(_p), 1, 2), IfThenElse(_p, 3, 4)),
     IfThenElse(_p, Block(2, 3), Block(1, 4))),

    (IfThen(_p, IfThen(_p, 1)),
     IfThen(_p, 1)),

    (IfThen(LogicalNot(_p), IfThen(LogicalNot(_p), 1)),
     IfThen(LogicalNot(_p), 1)),

    (IfThen(_p, IfThen(LogicalNot(_p), 1)),
     Block()),

    (IfThen(LogicalNot(_p), IfThen(_p, 1)),
     Block()),

    (IfThenElse(_p, IfThen(_p, 1), 2),
     IfThenElse(_p, 1, 2)),

    (IfThenElse(_p, IfThen(LogicalNot(_p), 1), 2),
     IfThen(LogicalNot(_p), 2)),

    (IfThen(True, 1),
     1),

    (IfThen(False, 1),
     Block()),

    (IfThenElse(True, 1, 2),
     1),

    (IfThenElse(False, 1, 2),
     2)
])
def test_simplify(in_ast, out_ast):
    assert simplify_ast(in_ast) == out_ast


# {{{ parser

def test_parser():
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
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
