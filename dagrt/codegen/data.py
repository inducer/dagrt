"""Mini-type inference for dagrt methods"""
from __future__ import division, with_statement, print_function

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

from dagrt.utils import TODO

import six
import dagrt.language as lang
from dagrt.utils import is_state_variable
from pytools import RecordWithoutPickling
from pymbolic.mapper import Mapper


def _get_arg_dict_from_call_insn(insn):
    arg_dict = {}
    for i, arg_val in enumerate(insn.parameters):
        arg_dict[i] = arg_val
    for arg_name, arg_val in insn.kw_parameters.items():
        arg_dict[arg_name] = arg_val

    return arg_dict


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

    def __repr__(self):
        return "%s(%s)" % (
                type(self).__name__,
                ", ".join(repr(arg) for arg in self.__getinitargs__()))

    def __hash__(self):
        return hash((type(self), self.__getinitargs__()))


class Boolean(SymbolKind):
    pass


class Integer(SymbolKind):
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


class Array(SymbolKind):
    """A variable-sized one-dimensional array.

    .. attribute:: is_real_valued

        Whether the value is definitely real-valued
    """

    def __init__(self, is_real_valued):
        super(Array, self).__init__(is_real_valued=is_real_valued)

    def __getinitargs__(self):
        return (self.is_real_valued,)


class UserType(SymbolKind):
    def __init__(self, identifier):
        super(UserType, self).__init__(identifier=identifier)

    def __getinitargs__(self):
        return (self.identifier,)

# }}}


class SymbolKindTable(object):
    """
    .. attribute:: global_table

        a mapping from symbol names to :class:`SymbolKind` instances,
        for global symbols

    .. attribute:: a nested mapping ``[function][symbol_name]``
        to :class:`SymbolKind` instances
    """

    def __init__(self):
        self.global_table = {
                "<t>": Scalar(is_real_valued=True),
                "<dt>": Scalar(is_real_valued=True),
                }
        self.per_function_table = {}
        self._changed = False

    def reset_change_flag(self):
        self._changed = False

    def is_changed(self):
        return self._changed

    def set(self, func_name, name, kind):
        if is_state_variable(name):
            tbl = self.global_table
        else:
            tbl = self.per_function_table.setdefault(func_name, {})

        if name in tbl:
            if tbl[name] != kind:
                try:
                    kind = unify(kind, tbl[name])
                except Exception:
                    print(
                        "trying to derive 'kind' for '%s' in "
                        "'%s': '%s' vs '%s'"
                        % (name, func_name,
                            repr(kind),
                            repr(tbl[name])))
                else:
                    if tbl[name] != kind:
                        self._changed = True
                        tbl[name] = kind

        else:
            tbl[name] = kind

    def get(self, func_name, name):
        if is_state_variable(name):
            tbl = self.global_table
        else:
            tbl = self.per_function_table.setdefault(func_name, {})

        return tbl[name]

    def __str__(self):
        def format_table(tbl, indent="  "):
            return "\n".join(
                    "%s%s: %s" % (indent, name, kind)
                    for name, kind in tbl.items())

        return "\n".join(
                ["global:\n%s" % format_table(self.global_table)] + [
                    "func '%s':\n%s" % (func_name, format_table(tbl))
                    for func_name, tbl in self.per_function_table.items()])


# {{{ kind inference mapper

class UnableToInferKind(Exception):
    pass


def unify(kind_a, kind_b):
    if kind_a is None:
        return kind_b
    if kind_b is None:
        return kind_a

    if isinstance(kind_a, Boolean):
        raise ValueError("arithmetic with flags is not permitted")
    if isinstance(kind_b, Boolean):
        raise ValueError("arithmetic with flags is not permitted")

    if isinstance(kind_a, UserType):
        assert isinstance(kind_b, (UserType, Scalar))

        if isinstance(kind_b, UserType):
            if kind_a.identifier != kind_b.identifier:
                raise ValueError(
                        "encountered arithmetic with mismatched "
                        "user types")

        return kind_a

    if isinstance(kind_a, Array):
        assert isinstance(kind_b, (Array, Scalar))

        return Array(
                not (not kind_a.is_real_valued or not kind_b.is_real_valued))

    elif isinstance(kind_a, Scalar):
        if isinstance(kind_b, UserType):
            return kind_b
        if isinstance(kind_b, Array):
            return Array(
                    not (not kind_a.is_real_valued or not kind_b.is_real_valued))
        if isinstance(kind_b, Integer):
            return kind_a

        assert isinstance(kind_b, Scalar)
        return Scalar(
                not (not kind_a.is_real_valued or not kind_b.is_real_valued))

    elif isinstance(kind_a, Integer):
        if isinstance(kind_b, (UserType, Scalar, Array)):
            return kind_b

        assert isinstance(kind_b, Integer)
        return Integer()

    raise NotImplementedError("unknown kind '%s'" % type(kind_a).__name__)


class KindInferenceMapper(Mapper):
    """
    .. attribute:: global_table

        The :class:`SymbolKindTable` for the global scope.

    .. attribute:: local_table

        The :class:`SymbolKindTable` for the :class:`dagrt.ir.Function`
        currently being processed.
    """

    def __init__(self, global_table, local_table, function_registry, check):
        self.global_table = global_table
        self.local_table = local_table
        self.function_registry = function_registry
        self.check = check

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
            return self.local_table[expr.name]
        except KeyError:
            pass

        raise UnableToInferKind(
                "nothing known about '%s'"
                % expr.name)

    def map_sum(self, expr):
        kind = None

        last_exc = None

        # Sums must be homogeneous, so being able to
        # infer one child is good enough.
        for ch in expr.children:
            try:
                ch_kind = self.rec(ch)
            except UnableToInferKind as e:
                if self.check:
                    raise
                else:
                    last_exc = e

            else:
                kind = unify(kind, ch_kind)

        if kind is None:
            raise last_exc
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
        if self.check and not isinstance(self.rec(expr.exponent), Scalar):
            raise TypeError(
                    "exponentiation by '%s'"
                    "is meaningless"
                    % type(self.rec(expr.exponent)).__name__)

    def map_generic_call(self, function_id, arg_dict, single_return_only=True):
        func = self.function_registry[function_id]
        arg_kinds = {}
        for key, val in arg_dict.items():
            try:
                arg_kinds[key] = self.rec(val)
            except UnableToInferKind:
                arg_kinds[key] = None

        z = func.get_result_kinds(arg_kinds, self.check)

        if single_return_only:
            if len(z) != 1:
                raise RuntimeError("Function '%s' is being used in an "
                        "expression context where it must return exactly "
                        "one value. It returned %d instead."
                        % (function_id, len(z)))
            return z[0]

        else:
            return z

    def map_call(self, expr):
        return self.map_generic_call(expr.function.name,
                dict(enumerate(expr.parameters)))

    def map_call_with_kwargs(self, expr):
        arg_dict = dict(enumerate(expr.parameters))
        arg_dict.update(expr.kw_parameters)
        return self.map_generic_call(expr.function.name, arg_dict)

    def map_comparison(self, expr):
        return Boolean()

    def map_logical_or(self, expr):
        for ch in expr.children:
            ch_kind = self.rec(ch)
            if self.check and not isinstance(ch_kind, Boolean):
                raise ValueError(
                        "logical operations on '%s' are undefined"
                        % type(ch_kind).__name__)

        return Boolean()

    map_logical_and = map_logical_or

    def map_logical_not(self, expr):
        ch_kind = self.rec(expr.child)
        if self.check and not isinstance(ch_kind, Boolean):
            raise ValueError(
                    "logical operations on '%s' are undefined"
                    % type(ch_kind).__name__)

        return Boolean()

    def map_max(self, expr):
        return Scalar(is_real_valued=True)

    map_min = map_max

    def map_subscript(self, expr):
        agg_kind = self.rec(expr.aggregate)
        if self.check and not isinstance(agg_kind, Array):
            raise ValueError(
                    "only arrays can be subscripted, not '%s' "
                    "which is a '%s'"
                    % (expr.aggregate, type(agg_kind).__name__))

        return Scalar(is_real_valued=agg_kind.is_real_valued)

# }}}


# {{{ symbol kind finder

class SymbolKindFinder(object):
    def __init__(self, function_registry):
        self.function_registry = function_registry

    def __call__(self, names, functions):
        """Return a :class:`SymbolKindTable`.
        """

        result = SymbolKindTable()

        from dagrt.codegen.ast import get_instructions_in_ast

        def make_kim(func_name, check):
            return KindInferenceMapper(
                    result.global_table,
                    result.per_function_table.get(func_name, {}),
                    self.function_registry,
                    check=False)

        while True:
            insn_queue = []
            for name, func in zip(names, functions):
                insn_queue.extend(
                        (name, insn)
                        for insn in get_instructions_in_ast(func))

            insn_queue_push_buffer = []
            made_progress = False

            result.reset_change_flag()

            while insn_queue or insn_queue_push_buffer:
                if not insn_queue:
                    # {{{ provide a usable error message if no progress

                    if not made_progress:
                        print("Left-over instructions in kind inference:")
                        for func_name, insn in insn_queue_push_buffer:
                            print("[%s] %s" % (func_name, insn))

                            kim = make_kim(func_name, check=False)

                            try:
                                if isinstance(insn, lang.AssignExpression):
                                    kim(insn.expression)

                                elif isinstance(insn, lang.AssignFunctionCall):
                                    kim.map_generic_call(insn.function_id,
                                            _get_arg_dict_from_call_insn(insn),
                                            single_return_only=False)

                                elif isinstance(insn, lang.AssignmentBase):
                                    raise TODO()

                                else:
                                    pass
                            except UnableToInferKind as e:
                                print("  -> %s" % str(e))
                            else:
                                # We aren't supposed to get here. Kind inference
                                # didn't succeed earlier. Since we made no progress,
                                # it shouldn't succeed now.
                                assert False

                        raise RuntimeError("failed to infer kinds")

                    # }}}

                    insn_queue = insn_queue_push_buffer
                    insn_queue_push_buffer = []
                    made_progress = False

                func_name, insn = insn_queue.pop()

                if isinstance(insn, lang.AssignExpression):
                    kim = make_kim(func_name, check=False)

                    for ident, _, _ in insn.loops:
                        result.set(func_name, ident, kind=Integer())

                    if insn.assignee_subscript:
                        continue

                    try:
                        kind = kim(insn.expression)
                    except UnableToInferKind:
                        insn_queue_push_buffer.append((func_name, insn))
                    else:
                        made_progress = True
                        result.set(func_name, insn.assignee, kind=kind)

                elif isinstance(insn, lang.AssignFunctionCall):
                    kim = make_kim(func_name, check=False)

                    try:
                        kinds = kim.map_generic_call(insn.function_id,
                                _get_arg_dict_from_call_insn(insn),
                                single_return_only=False)
                    except UnableToInferKind:
                        insn_queue_push_buffer.append((func_name, insn))
                    else:
                        made_progress = True
                        for assignee, kind in zip(insn.assignees, kinds):
                            result.set(func_name, assignee, kind=kind)

                elif isinstance(insn, lang.AssignmentBase):
                    raise TODO()

                else:
                    # We only care about assignments.
                    pass

            if not result.is_changed():
                break

        # {{{ check consistency of obtained kinds

        for func_name, func in zip(names, functions):
            kim = make_kim(func_name, check=True)

            for insn in get_instructions_in_ast(func):
                if isinstance(insn, lang.AssignExpression):
                    kim(insn.expression)

                elif isinstance(insn, lang.AssignFunctionCall):
                    kim.map_generic_call(insn.function_id,
                            _get_arg_dict_from_call_insn(insn),
                            single_return_only=False)

                    func = self.function_registry[insn.function_id]
                    if len(func.result_names) != len(insn.assignees):
                        raise ValueError("number of function return values "
                                "for '%s' (%d) "
                                "and number of assigned variables (%d) "
                                "do not match"
                                % (insn.function_id,
                                    len(func.result_names), len(insn.assignees)))

                elif isinstance(insn, lang.AssignmentBase):
                    raise TODO()

                else:
                    pass

        # }}}

        return result

# }}}


# {{{ collect user types

def collect_user_types(skt):
    result = set()

    for kind in six.itervalues(skt.global_table):
        if isinstance(kind, UserType):
            result.add(kind.identifier)

    for tbl in six.itervalues(skt.per_function_table):
        for kind in six.itervalues(tbl):
            if isinstance(kind, UserType):
                result.add(kind.identifier)

    return result

# }}}

# vim: foldmethod=marker
