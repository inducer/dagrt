"""Mini-type inference for dagrt methods"""

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

import dagrt.language as lang
from dagrt.utils import is_state_variable
from pytools import RecordWithoutPickling
from pymbolic.mapper import Mapper


__doc__ = """
The module :mod:`dagrt.data` provides symbol kind information and kind
inference.

Symbol kinds
^^^^^^^^^^^^

.. autoclass:: SymbolKind
.. autoclass:: Boolean
.. autoclass:: Integer
.. autoclass:: Scalar
.. autoclass:: Array
.. autoclass:: UserType

Symbol kind inference
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SymbolKindTable
.. autoclass:: UnableToInferKind
.. autoclass:: SymbolKindFinder

Helper functions
^^^^^^^^^^^^^^^^

.. autofunction:: infer_kinds
.. autofunction:: collect_user_types

"""


def _get_arg_dict_from_call_stmt(stmt):
    arg_dict = {}
    for i, arg_val in enumerate(stmt.parameters):
        arg_dict[i] = arg_val
    for arg_name, arg_val in stmt.kw_parameters.items():
        arg_dict[arg_name] = arg_val

    return arg_dict


# {{{ symbol information

class SymbolKind(RecordWithoutPickling):
    """Base class for kinds encountered in the :mod:`dagrt` language."""

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.__getinitargs__() == other.__getinitargs__())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self.__getinitargs__()))

    def __getinitargs__(self):
        return ()


class Boolean(SymbolKind):
    """A boolean."""
    pass


class Integer(SymbolKind):
    """An integer."""
    pass


class Scalar(SymbolKind):
    """A (real or complex) floating-point value.

    .. attribute:: is_real_valued

        Whether the value is definitely real-valued
    """

    def __init__(self, is_real_valued):
        super().__init__(is_real_valued=is_real_valued)

    def __getinitargs__(self):
        return (self.is_real_valued,)


class Array(SymbolKind):
    """A variable-sized one-dimensional scalar array.

    .. attribute:: is_real_valued

        Whether the value is definitely real-valued
    """

    def __init__(self, is_real_valued):
        super().__init__(is_real_valued=is_real_valued)

    def __getinitargs__(self):
        return (self.is_real_valued,)


class UserType(SymbolKind):
    """Represents user state belonging to a normed vector space.

    .. attribute:: identifier

        A unique identifier for this type
    """

    def __init__(self, identifier):
        super().__init__(identifier=identifier)

    def __getinitargs__(self):
        return (self.identifier,)

# }}}


class SymbolKindTable:
    """A mapping from symbol names to kinds for a program.

    .. attribute:: global_table

        a mapping from symbol names to :class:`SymbolKind` instances,
        for global symbols

    .. attribute:: per_phase_table

        a nested mapping ``[phase_name][symbol_name]`` to :class:`SymbolKind`
        instances
    """

    def __init__(self):
        self.global_table = {
                "<t>": Scalar(is_real_valued=True),
                "<dt>": Scalar(is_real_valued=True),
                }
        self.per_phase_table = {}
        self._changed = False

    def reset_change_flag(self):
        self._changed = False

    def is_changed(self):
        return self._changed

    def set(self, phase_name, name, kind):
        if is_state_variable(name):
            tbl = self.global_table
        else:
            tbl = self.per_phase_table.setdefault(phase_name, {})

        if name in tbl:
            if tbl[name] != kind:
                try:
                    kind = unify(kind, tbl[name])
                except Exception:
                    print(
                        "trying to derive 'kind' for '%s' in "
                        "'%s': '%s' vs '%s'"
                        % (name, phase_name,
                            repr(kind),
                            repr(tbl[name])))
                else:
                    if tbl[name] != kind:
                        self._changed = True
                        tbl[name] = kind

        else:
            tbl[name] = kind

    def get(self, phase_name, name):
        if is_state_variable(name):
            tbl = self.global_table
        else:
            tbl = self.per_phase_table.setdefault(phase_name, {})

        return tbl[name]

    def __str__(self):
        def format_table(tbl, indent="  "):
            return "\n".join(
                    f"{indent}{name}: {kind}"
                    for name, kind in tbl.items())

        return "\n".join(
                ["global:\n%s" % format_table(self.global_table)] + [
                    "phase '{}':\n{}".format(phase_name, format_table(tbl))
                    for phase_name, tbl in self.per_phase_table.items()])


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

        The :class:`SymbolKindTable` for the phase currently being processed.
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

class SymbolKindFinder:
    """
    .. automethod:: __call__
    """
    def __init__(self, function_registry):
        self.function_registry = function_registry

    def __call__(self, names, phases, forced_kinds=None):
        """Infer the kinds of all the symbols in a program.

        :arg names: a list of phase names
        :arg phases: a list of iterables, each yielding the statements in a
            phase

        :returns: a :class:`SymbolKindTable`

        :raises UnableToInferKind: kind inference could not complete sucessfully
        """

        expanded_phases = []
        for phase in phases:
            expanded_phases.append(list(phase))
        phases = expanded_phases

        result = SymbolKindTable()

        if forced_kinds is not None:
            for phase_name, ident, kind in forced_kinds:
                result.set(phase_name, ident, kind=kind)

        def make_kim(phase_name, check):
            return KindInferenceMapper(
                    result.global_table,
                    result.per_phase_table.get(phase_name, {}),
                    self.function_registry,
                    check=False)

        while True:
            stmt_queue = []
            for name, phase in zip(names, phases):
                stmt_queue.extend((name, stmt) for stmt in phase)

            stmt_queue_push_buffer = []
            made_progress = False

            result.reset_change_flag()

            while stmt_queue or stmt_queue_push_buffer:
                if not stmt_queue:
                    # {{{ provide a usable error message if no progress

                    if not made_progress:
                        print("Left-over statements in kind inference:")
                        for phase_name, stmt in stmt_queue_push_buffer:
                            print(f"[{phase_name}] {stmt}")

                            kim = make_kim(phase_name, check=False)

                            try:
                                if isinstance(stmt, lang.Assign):
                                    kim(stmt.expression)

                                elif isinstance(stmt, lang.AssignFunctionCall):
                                    kim.map_generic_call(stmt.function_id,
                                            _get_arg_dict_from_call_stmt(stmt),
                                            single_return_only=False)

                                elif isinstance(stmt, lang.AssignmentBase):
                                    raise TODO()

                                else:
                                    pass
                            except UnableToInferKind as e:
                                print("  -> %s" % str(e))
                            else:
                                # We aren't supposed to get here. Kind inference
                                # didn't succeed earlier. Since we made no progress,
                                # it shouldn't succeed now.
                                raise AssertionError()

                        raise RuntimeError("failed to infer kinds")

                    # }}}

                    stmt_queue = stmt_queue_push_buffer
                    stmt_queue_push_buffer = []
                    made_progress = False

                phase_name, stmt = stmt_queue.pop()

                if isinstance(stmt, lang.Assign):
                    kim = make_kim(phase_name, check=False)

                    for ident, _, _ in stmt.loops:
                        result.set(phase_name, ident, kind=Integer())

                    if stmt.assignee_subscript:
                        continue

                    try:
                        kind = kim(stmt.expression)
                    except UnableToInferKind:
                        stmt_queue_push_buffer.append((phase_name, stmt))
                    else:
                        made_progress = True
                        result.set(phase_name, stmt.assignee, kind=kind)

                elif isinstance(stmt, lang.AssignFunctionCall):
                    kim = make_kim(phase_name, check=False)

                    try:
                        kinds = kim.map_generic_call(stmt.function_id,
                                _get_arg_dict_from_call_stmt(stmt),
                                single_return_only=False)
                    except UnableToInferKind:
                        stmt_queue_push_buffer.append((phase_name, stmt))
                    else:
                        made_progress = True
                        for assignee, kind in zip(stmt.assignees, kinds):
                            result.set(phase_name, assignee, kind=kind)

                elif isinstance(stmt, lang.AssignmentBase):
                    raise TODO()

                else:
                    # We only care about assignments.
                    pass

            if not result.is_changed():
                break

        # {{{ check consistency of obtained kinds

        for phase_name, phase in zip(names, phases):
            kim = make_kim(phase_name, check=True)

            for stmt in phase:
                if isinstance(stmt, lang.Assign):
                    kim(stmt.expression)

                elif isinstance(stmt, lang.AssignFunctionCall):
                    kim.map_generic_call(stmt.function_id,
                            _get_arg_dict_from_call_stmt(stmt),
                            single_return_only=False)

                    func = self.function_registry[stmt.function_id]
                    if len(func.result_names) != len(stmt.assignees):
                        raise ValueError("number of function return values "
                                "for '%s' (%d) "
                                "and number of assigned variables (%d) "
                                "do not match"
                                % (stmt.function_id,
                                    len(func.result_names), len(stmt.assignees)))

                elif isinstance(stmt, lang.AssignmentBase):
                    raise TODO()

                else:
                    pass

        # }}}

        return result

# }}}


# {{{ infer kinds of a DAGCode object

def infer_kinds(dag, function_registry=None):
    """Run kind inference on a :class:`dagrt.language.DAGCode`.

    :arg dag: a :class:`dagrt.language.DAGCode`
    :arg function_registry: if not *None*, the function registry to use

    :returns: a :class:`SymbolKindTable`
    """
    if function_registry is None:
        from dagrt.function_registry import base_function_registry
        function_registry = base_function_registry

    kind_finder = SymbolKindFinder(function_registry)
    names = list(dag.phases)
    phases = [phase.statements for phase in dag.phases.values()]

    return kind_finder(names, phases)

# }}}


# {{{ collect user types

def collect_user_types(skt):
    """Collect all of the of :class:`UserType` identifiers in a table.

    :arg skt: a :class:`SymbolKindTable`
    :returns: a set of strings
    """
    result = set()

    for kind in skt.global_table.values():
        if isinstance(kind, UserType):
            result.add(kind.identifier)

    for tbl in skt.per_phase_table.values():
        for kind in tbl.values():
            if isinstance(kind, UserType):
                result.add(kind.identifier)

    return result

# }}}

# vim: foldmethod=marker
