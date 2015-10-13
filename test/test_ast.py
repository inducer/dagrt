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

from pymbolic.primitives import LogicalNot

from dagrt.vm.codegen.ast import (IfThen, IfThenElse, Block, InstructionWrapper,
                                 match_ast, declare, create_ast_from_state,
                                 simplify_ast)
from dagrt.vm.language import Nop, TimeIntegratorCode


def test_create_ast():
    x, = declare("x")

    class ComparableNop(Nop):
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
    code = TimeIntegratorCode.create_with_steady_state(["nop"], [nop])
    ast = create_ast_from_state(code, "main", simplify=False)
    assert ast == Block(IfThen(x, InstructionWrapper(nop.copy(condition=True))))


def test_match():
    x, y = declare("x", "y")

    def pass_match(expression, template, *matches):
        match_result = match_ast(expression, template)
        assert match_result == dict(matches)

    def fail_match(expression, template):
        assert match_ast(expression, template) is None

    pass_match(IfThen(0, 1), IfThen(0, 1))
    pass_match(IfThen(0, 1), IfThen(x, 1), ("x", 0))
    pass_match(IfThen(0, 1), IfThen(x, y), ("x", 0), ("y", 1))
    pass_match(IfThen(0, 0), IfThen(x, x), ("x", 0))

    fail_match(IfThen(0, 1), IfThen(x, x))
    fail_match(IfThen(0, 1), IfThen(0, 2))

    pass_match(IfThen(LogicalNot(7), 0), IfThen(LogicalNot(x), 0), ("x", 7))
    fail_match(IfThen(7, 0), IfThen(LogicalNot(x), 0))

    pass_match(IfThenElse(0, 1, 2), IfThenElse(x, 1, 2), ("x", 0))
    pass_match(IfThenElse(0, 1, 2), IfThenElse(0, 1, y), ("y", 2))
    pass_match(IfThenElse(0, 1, 1), IfThenElse(x, y, y), ("x", 0), ("y", 1))
    pass_match(IfThenElse(LogicalNot(0), 1, 2), IfThenElse(x, 1, 2),
               ("x", LogicalNot(0)))

    fail_match(IfThenElse(0, 1, 2), IfThenElse(x, y, y))
    fail_match(IfThenElse(0, 1, 2), IfThen(0, 1))

    pass_match(Block(1), Block(1))
    pass_match(Block(IfThen(0, 1), IfThen(0, 2)),
               Block(IfThen(x, 1), IfThen(x, 2)),
               ("x", 0))


def test_simplify():
    p, = declare("p")
    assert simplify_ast(Block(Block(0))) == 0
    assert simplify_ast(Block(IfThen(p, 0), IfThen(p, 1), IfThen(p, 2))) == \
        IfThen(p, Block(0, 1, 2))
    assert simplify_ast(Block(IfThen(p, 0), IfThen(LogicalNot(p), 1))) == \
        IfThenElse(p, 0, 1)
    assert simplify_ast(Block(IfThen(p, 0), IfThenElse(p, 1, 2))) == \
        IfThenElse(p, Block(0, 1), 2)
    assert simplify_ast(Block(IfThenElse(p, 0, 1), IfThen(LogicalNot(p), 2))) == \
        IfThenElse(p, 0, Block(1, 2))

    # Check that simplification respects redefinitions.
    from dagrt.vm.language import AssignExpression
    redef = InstructionWrapper(AssignExpression("p", (), 10))
    input_same_as_output = lambda f, x: f(x) == x
    assert input_same_as_output(simplify_ast, Block(IfThen(p, redef), IfThen(p, 0)))
    assert input_same_as_output(simplify_ast, Block(IfThenElse(p, 0, redef),
                                                    IfThen(LogicalNot(p), 1)))
    assert input_same_as_output(simplify_ast, Block(IfThenElse(p, redef, 0),
                                                    IfThen(LogicalNot(p), 1)))
    assert input_same_as_output(simplify_ast, Block(IfThen(p, redef),
                                                    IfThenElse(p, 0, 1)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
