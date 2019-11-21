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

from dagrt.codegen.dag_ast import (IfThen, IfThenElse, Block, StatementWrapper,
                               create_ast_from_phase, simplify_ast)
from dagrt.language import Statement

from utils import create_DAGCode_with_steady_phase

import pytest


def test_create_ast():
    x = var("<cond>x")

    class ComparableNop(Statement):
        """
        Works around the fact that __hash__ and comparison isn't defined for
        statements, but is necessary for comparison purposes when embedded
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
    code = create_DAGCode_with_steady_phase([nop])
    ast = create_ast_from_phase(code, "main")
    assert ast == IfThen(x, StatementWrapper(nop.copy(condition=True)))


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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
