"""Mini-type inference for leap methods"""

from __future__ import division, with_statement

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2014 Matt Wala
"""

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

import leap.vm.language as lang
import leap.vm.codegen.ir as ir
from pytools import RecordWithoutPickling
from pymbolic.mapper import Mapper


# {{{ symbol information

class SymbolKind(RecordWithoutPickling):
    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.__getinitargs__() == other.__getinitargs__())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getinitargs__(self):
        return ()


class Flag(SymbolKind):
    pass


class Scalar(SymbolKind):
    """
    .. attribute:: is_real_valued

        Whether the value is definitely real-valued
    """

    def __init__(self, is_real_valued):
        super(Scalar, self).__init__(is_real_valued=is_real_valued)

    def __getinitargs__(self):
        return (self.is_real_valued,)


class ODEComponent(SymbolKind):
    def __init__(self, component_id):
        super(ODEComponent, self).__init__(component_id=component_id)

    def __getinitargs__(self):
        return (self.component_id,)

# }}}


class SymbolKindTable(object):
    """
    .. attribute:: global_table

        a mapping from symbol names to :class:`SymbolKind` instances,
        for global symbols

    .. attribute:: per_function_table

        a mapping from ``(function, symbol_name)``
        to :class:`SymbolKind` instances
    """

    def __init__(self):
        self.global_table = {
                "<t>": Scalar(is_real_valued=True),
                "<dt>": Scalar(is_real_valued=True),
                }
        self.per_function_table = {}

    def _set(self, func_name, name, kind):
        from leap.vm.utils import is_state_variable
        if is_state_variable(name):
            tbl = self.global_table
        else:
            tbl = self.per_function_table.setdefault(func_name, {})

        if name in tbl:
            if tbl[name] != kind:
                raise RuntimeError(
                        "inconsistent 'kind' derived for '%s' in "
                        "'%s': '%s' vs '%s'"
                        % (name, func_name,
                            type(kind).__name__,
                            type(tbl[name]).__name__))
        else:
            tbl[name] = kind

    def __str__(self):
        def format_table(tbl, indent="  "):
            return "\n".join(
                    "%s%s: %s" % (indent, name, kind)
                    for name, kind in tbl.items())

        return "\n".join(
                ["global:\n%s" % format_table(self.global_table)] + [
                    "func '%s':\n%s" % (func_name, format_table(tbl))
                    for func_name, tbl in self.per_function_table.items()])


# {{{ type inference mapper

class UnableToInferType(Exception):
    pass


def unify(kind_a, kind_b):
    if kind_a is None:
        return kind_b
    if kind_b is None:
        return kind_a

    if isinstance(kind_a, Flag):
        raise ValueError("arithmetic with flags is not permitted")
    if isinstance(kind_b, Flag):
        raise ValueError("arithmetic with flags is not permitted")

    if isinstance(kind_a, ODEComponent):
        assert isinstance(kind_b, (ODEComponent, Scalar))

        if isinstance(kind_b, ODEComponent):
            if kind_a.component_id != kind_b.component_id:
                raise ValueError(
                        "encountered arithmetic with mismatched "
                        "ODE components")

        return kind_a

    elif isinstance(kind_a, Scalar):
        if isinstance(kind_b, ODEComponent):
            return kind_b

        assert isinstance(kind_b, Scalar)
        return Scalar(
                not (not kind_a.is_real_valued or not kind_b.is_real_valued))

    raise NotImplementedError("unknown kind '%s'" % type(kind_a).__name__)


class KindInferenceMapper(Mapper):
    def __init__(self, global_table, func_table):
        self.global_table = global_table
        self.func_table = func_table

    def map_constant(self, expr):
        if isinstance(expr, complex):
            return Scalar(is_real_valued=False)
        else:
            return Scalar(is_real_valued=True)

    def map_variable(self, expr):
        try:
            return self.global_table[expr.name]
        except KeyError:
            pass

        try:
            return self.func_table[expr.name]
        except KeyError:
            pass

        raise UnableToInferType()

    def map_sum(self, expr):
        kind = None
        for ch in expr.children:
            try:
                ch_kind = self.rec(ch)
            except UnableToInferType:
                pass
            else:
                kind = unify(kind, ch_kind)

        if kind is None:
            raise UnableToInferType()
        else:
            return kind

    def map_product_like(self, children):
        kind = None
        for ch in children:
            kind = unify(kind, self.rec(ch))

        return kind

    def map_product(self, expr):
        return self.map_product_like(expr.children)

    def map_quotient(self, expr):
        return self.map_product_like((expr.numerator, expr.denominator))

    def map_power(self, expr):
        if not isinstance(self.rec(expr.exponent), Scalar):
            raise ValueError(
                    "exponentiation by '%s'"
                    "is meaningless"
                    % type(self.rec(expr.exponent)).__name__)

# }}}


# {{{ symbol kind finder

class SymbolKindFinder(object):
    def __call__(self, functions):
        """Return a :class:`SymbolKindTable`.
        """

        result = SymbolKindTable()

        insn_queue = [
                (func.name, insn)
                for func in functions
                for bblock in func.postorder()
                for insn in bblock]
        insn_queue_push_buffer = []
        made_progress = False

        while insn_queue or insn_queue_push_buffer:
            if not insn_queue:
                if not made_progress:
                    raise RuntimeError("failed to infer types")

                insn_queue = insn_queue_push_buffer
                insn_queue_push_buffer = []

            func_name, insn = insn_queue.pop()

            if isinstance(insn, ir.AssignInst):
                if isinstance(insn.assignment, lang.AssignRHS):
                    for name in insn.assignment.assignees:
                        made_progress = True
                        result._set(
                                func_name, name,
                                kind=ODEComponent(insn.assignment.component_id))

                elif isinstance(insn.assignment, lang.AssignSolvedRHS):
                    made_progress = True
                    from leap.vm.utils import TODO
                    raise TODO()

                elif isinstance(insn.assignment, lang.AssignDotProduct):
                    made_progress = True
                    result._set(
                            func_name, insn.assignment.assignee,
                            kind=Scalar(is_real_valued=False))

                elif isinstance(insn.assignment, lang.AssignNorm):
                    made_progress = True
                    result._set(
                            func_name, insn.assignment.assignee,
                            kind=Scalar(is_real_valued=True))

                elif isinstance(insn.assignment, tuple):
                    # These are function returns or initializations, we're not
                    # interested in them.

                    name, expr = insn.assignment
                    assert expr is None or isinstance(expr, tuple)

                elif isinstance(insn.assignment, (tuple, lang.AssignExpression)):
                    kim = KindInferenceMapper(
                            result.global_table,
                            result.per_function_table.get(func_name, {}))
                    try:
                        kind = kim(insn.assignment.expression)
                    except UnableToInferType:
                        insn_queue_push_buffer.append((func_name, insn))
                    else:
                        made_progress = True
                        result._set(
                                func_name, insn.assignment.assignee,
                                kind=kind)

                else:
                    raise NotImplementedError(
                            "assignment of type '%s'"
                            % type(insn.assignment).__name__)

            else:
                # We only care about assignments.
                pass

        return result

# }}}

# vim: foldmethod=marker
