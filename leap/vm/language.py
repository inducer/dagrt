from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

from pytools import Record, memoize_method

import logging
logger = logging.getLogger(__name__)


# {{{ utilities

def get_dependencies(expr):
    from pymbolic.mapper.dependency import DependencyMapper
    dep_mapper = DependencyMapper(composite_leaves=False)

    return frozenset(dep.name for dep in dep_mapper(expr))

# }}}


# {{{ instructions

__doc__ = """
Identifier conventions
~~~~~~~~~~~~~~~~~~~~~~

Identifiers whose names start with the pattern <letters> are special.  The
following special variable names are supported:

``<p>NAME``
    This variable contains persistent state.
    (that survives from one step to the next)

``<dt>``
    The time increment for the present time step.

    Its value at the beginning of a step indicates the step size to be used. If
    a time step of this size cannot be completed, FailStep must be issued.

    This variable contains persistent state.
    (that survives from one step to the next)

``<t>``
    Base time of current time step.

    The integrator code is responsible for incrementing <t> at the end of a
    successful step.

    This variable contains persistent state.
    (that survives from one step to the next)

``<state>NAME``
    State identifier under user control

    This variable contains persistent state.
    (that survives from one step to the next)

``<func>NAME``
    A user-defined function.

The latter two serve to separate the name space used by the method from that
under the control of the user.

Built-in functions:

* ``len(state)`` returns the number of degrees of freedom in *state*
* ``isnan(state)`` returns True if there are any NaNs in *state*
"""


class Instruction(Record):
    """
    .. attribute:: id

        A string, a unique identifier for this instruction.

    .. attribute:: depends_on

        A :class:`frozenset` of instruction ids that are reuqired to be
        executed within this execution context before this instruction can be
        executed.

    .. automethod:: get_assignees
    .. automethod:: get_read_variables
    """

    def __init__(self, **kwargs):
        id = kwargs.pop("id", None)
        if id is not None:
            id = intern(id)
        depends_on = frozenset(kwargs.pop("depends_on", []))
        Record.__init__(self,
                id=id,
                depends_on=depends_on,
                **kwargs)

    def get_assignees(self):
        """Returns a :class:`frozenset` of variables being written by this
        instruction.
        """
        raise NotImplementedError()

    def get_read_variables(self):
        """Returns a :class:`frozenset` of variables being read by this
        instruction.
        """
        raise NotImplementedError()


# {{{ assignments

class AssignRHS(Instruction):
    """
    .. attribute:: assignees

        A tuple of strings, the names of the variable being assigned to.

    .. attribute:: component_id

        Identifier of the right hand side to be evaluated. Typically a number
        or a string.

    .. attribute:: t

        A :mod:`pymbolic` expression for the time at which the right hand side
        is to be evaluated. Time is implicitly the first argument to each
        expression.

    .. attribute:: rhs_arguments

        A tuple of tuples. The outer tuple is the vectorization list, where each
        entry corresponds to one of the entries of :attr:`assignees`. The inner
        lists corresponds to arguments being passed to the right-hand side
        (identified by :attr:`component_id`) being invoked. These are tuples
        ``(arg_name, expression)``, where *expression* is a :mod:`pymbolic`
        expression.
    """

    def get_assignees(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        result = get_dependencies(self.t)
        for args in self.rhs_arguments:
            for name, expr in args:
                result = result | get_dependencies(expr)

        return result

    def __str__(self):
        lines = ["at time: %s" % self.t]
        for assignee, rhs_args in zip(self.assignees, self.rhs_arguments):
            lines.append("%s <- rhs:%s(%s)"
                    % (assignee, self.component_id,
                        ", ".join(str(arg)
                            for name, arg in rhs_args)))

        return "\n".join(lines)

    exec_method = intern("exec_AssignRHS")


class AssignSolvedRHS(Instruction):
    # FIXME: We should allow vectorization over multiple inputs/outputs
    # Pure vectorization is not enough here. We may want some amount
    # of tensor product expression flexibility.

    # FIXME This needs some thought.
    """
    .. attribute:: assignee

        A string, the name of the variable being assigned to.

    .. attribute:: component_id
    .. attribute:: t
    .. attribute:: states
    .. attribute:: solve_component
    .. attribute:: rhs
    """

    exec_method = intern("exec_AssignSolvedRHS")


class AssignExpression(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: expression
    """
    def __init__(self, assignee, expression, id=None, depends_on=frozenset()):
        Instruction.__init__(self,
                assignee=assignee,
                expression=expression,
                id=id,
                depends_on=depends_on)

    def get_assignees(self):
        return frozenset([self.assignee])

    def get_read_variables(self):
        return get_dependencies(self.expression)

    def __str__(self):
        return "%s <- %s" % (self.assignee, self.expression)

    exec_method = "exec_AssignExpression"


class AssignNorm(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: expression
    .. attribute:: p
    """

    def __init__(self, assignee, expression, p=2, id=None, depends_on=frozenset()):
        Instruction.__init__(self,
                assignee=assignee,
                expression=expression,
                p=p,
                id=id,
                depends_on=depends_on)

    def get_assignees(self):
        return frozenset([self.assignee])

    def get_read_variables(self):
        return get_dependencies(self.expression)

    def __str__(self):
        return "%s <- ||%s||_%s" % (self.assignee, self.expression, self.p)

    exec_method = "exec_AssignNorm"


class AssignDotProduct(Instruction):
    """
    .. attribute:: assignee
    .. attribute:: expression_1

        The complex conjugate of this argument is taken before computing the
        dot product, if applicable.

    .. attribute:: expression_2
    """

    def get_assignees(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        return (
                get_dependencies(self.expression_1)
                | get_dependencies(self.expression_2))

    exec_method = "exec_AssignDotProduct"

# }}}


class ReturnState(Instruction):
    """
    .. attribute:: time_id
    .. attribute:: time

        A :mod:`pymbolic` expression representing the time at which the
        state returned is valid.

    .. attribute:: component_id
    .. attribute:: expression
    """

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return get_dependencies(self.expression)

    def __str__(self):
        return "Ret %s at %s as %s" % (
                self.expression,
                self.time_id,
                self.component_id)

    exec_method = intern("exec_ReturnState")


class Raise(Instruction):
    """
    .. attribute:: error_condition

        A (Python) exception type to be raised.

    .. attribute:: error_message

        The error message to go along with that exception type.
    """

    def __init__(self, error_condition, error_message=None,
            id=None, depends_on=frozenset()):
        Instruction.__init__(self,
                error_condition=error_condition,
                error_message=error_message,
                id=id,
                depends_on=depends_on)

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return frozenset()

    def __str__(self):
        result = "Raise %s" % self.error_condition.__name__

        if self.error_message:
            result += repr(self.error_message)

        return result

    exec_method = "exec_Raise"


class FailStep(Instruction):
    """
    """

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return frozenset()

    def __str__(self):
        return "FailStep"

    exec_method = "exec_FailStep"


class If(Instruction):
    """
    .. attribute:: condition
    .. attribute:: then_depends_on

        a set of ids that the instruction depends on if :attr:`condition_expr`
        evaluates to True

    .. attribute:: else_depends_on

        a set of ids that the instruction depends on if :attr:`condition`
        evaluates to False
    """

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return get_dependencies(self.condition)

    def __str__(self):
        return "If %s" % self.condition

    exec_method = "exec_If"

# }}}

# light cone optimizations?


# {{{ code container

class TimeIntegratorCode(Record):
    """
    .. attribute:: initialization_dep_on

        List of instruction ids (not including their recursive dependencies) to
        be executed for initialization of the time integrator.

    .. attribute:: step_dep_on

        List of instruction ids (not including their recursive dependencies) to
        be executed for one time step.

    .. attribute:: instructions

        A list of :class:`Instruction` instances, in no particular order.

    .. attribute:: step_before_fail

        Whether the described method may generate state updates (using
        :class:`ReturnState`) for a time step it later decides to fail
        (using :class:`FailStep`).
    """

    def __init__(self,
            initialization_dep_on,
            step_dep_on,
            instructions,
            step_before_fail):
        Record.__init__(self,
                initialization_dep_on=initialization_dep_on,
                step_dep_on=step_dep_on,
                instructions=instructions,
                step_before_fail=step_before_fail)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn)
                for insn in self.instructions)

# }}}


# {{{ interpreter foundation

class ExecutionController(object):
    def __init__(self, code):
        self.code = code

        self.plan = []
        self.executed_ids = set()
        self.plan_id_set = set()

    def reset(self):
        logger.debug("execution reset")
        self.executed_ids.clear()

    def update_plan(self, execute_ids):
        """Update the plan with the minimal list of instruction ids to execute
        so that the instruction IDs in execute_ids will be executed before any
        others and such that all their dependencies are satisfied.
        """

        early_plan = []

        id_to_insn = self.code.id_to_insn

        def add_with_deps(insn):
            insn_id = insn.id
            if insn_id in self.executed_ids:
                # Already done, no need to think more.
                return

            if insn_id in self.plan_id_set:
                # Already in plan, no need to think more.
                return

            if insn_id in early_plan:
                return

            for dep_id in insn.depends_on:
                add_with_deps(id_to_insn[dep_id])

            assert insn_id not in self.plan_id_set

            early_plan.append(insn_id)

        for insn_id in execute_ids:
            add_with_deps(id_to_insn[insn_id])

        self.plan = early_plan + self.plan
        self.plan_id_set.update(early_plan)

    def __call__(self, target):
        id_to_insn = self.code.id_to_insn

        while self.plan:
            insn_id = self.plan.pop(0)
            self.plan_id_set.remove(insn_id)

            self.executed_ids.add(insn_id)

            insn = id_to_insn[insn_id]

            logger.debug("execution trace: [%s] %s" % (
                insn.id, str(insn).replace("\n", " | ")))

            result = getattr(target, insn.exec_method)(insn)
            if result is not None:
                event, new_deps = result
                if event is not None:
                    yield event

                if new_deps is not None:
                    self.update_plan(new_deps)

# }}}


# {{{ code building utility

class CodeBuilder(object):
    def __init__(self):
        self.id_set = set()
        self._instructions = []
        self.build_group = []

        from pytools import generate_unique_names
        self.id_generator = generate_unique_names("insn")

    def add_and_get_ids(self, *insns):
        new_ids = []
        for insn in insns:
            set_attrs = {}
            if not hasattr(insn, "id") or insn.id is None:

                for possible_id in self.id_generator:
                    if possible_id not in self.id_set:
                        set_attrs["id"] = possible_id
                        break
            else:
                if insn.id in self.id_set:
                    raise ValueError("duplicate ID")

            if not hasattr(insn, "depends_on"):
                set_attrs["depends_on"] = frozenset()

            if set_attrs:
                insn = insn.copy(**set_attrs)

            self.build_group.append(insn)
            new_ids.append(insn.id)

        # For exception safety, only make state change at end.
        self.id_set.update(new_ids)
        return new_ids

    def infer_single_writer_dependencies(self, exclude=[]):
        """Change the current :attr:`build_group` to one with reader
        dependencies added in situations where a only single instruction writes
        one variable.
        """

        var_to_writers = {}
        for insn in self.build_group:
            for wvar in insn.get_assignees():
                var_to_writers.setdefault(wvar, []).append(insn)

        # toss out variables written more than once
        for wvar in list(var_to_writers.iterkeys()):
            if len(var_to_writers[wvar]) > 1:
                del var_to_writers[wvar]

        # toss out excluded variables
        for v in exclude:
            try:
                del var_to_writers[v]
            except KeyError:
                pass

        new_build_group = []

        single_writer_vars = set(var_to_writers)

        for insn in self.build_group:
            new_deps = []
            for v in insn.get_read_variables() & single_writer_vars:
                var_writer, = var_to_writers[v]
                new_deps.append(var_writer.id)

            new_build_group.append(
                    insn.copy(
                        depends_on=insn.depends_on | frozenset(new_deps)))

        self.build_group = new_build_group

    def commit(self):
        self._instructions.extend(self.build_group)
        del self.build_group[:]

    @property
    def instructions(self):
        if self.build_group:
            raise ValueError("attempted to get instructions while "
                    "build group is uncommitted")

        return self._instructions

# }}}


# {{{ graphviz / dot export

def get_dot_dependency_graph(code, use_insn_ids=False):
    """Return a string in the `dot <http://graphviz.org/>`_ language depicting
    dependencies among kernel instructions.
    """

    lines = []
    dep_graph = {}

    # maps (oriented) edge onto annotation string
    annotation_dep_graph = {}

    for insn in code.instructions:
        if use_insn_ids:
            insn_label = insn.id
            tooltip = str(insn)
        else:
            insn_label = str(insn)
            tooltip = insn.id

        lines.append("\"%s\" [label=\"%s\",shape=\"box\",tooltip=\"%s\"];"
                % (
                    insn.id,
                    repr(insn_label)[1:-1],
                    repr(tooltip)[1:-1],
                    ))
        for dep in insn.depends_on:
            dep_graph.setdefault(insn.id, set()).add(dep)

        if isinstance(insn, If):
            for dep in insn.then_depends_on:
                annotation_dep_graph[(insn.id, dep)] = "then"
            for dep in insn.else_depends_on:
                annotation_dep_graph[(insn.id, dep)] = "else"

    # {{{ O(n^3) (i.e. slow) transitive reduction

    # first, compute transitive closure by fixed point iteration
    while True:
        changed_something = False

        for insn_1 in dep_graph:
            for insn_2 in dep_graph.get(insn_1, set()).copy():
                for insn_3 in dep_graph.get(insn_2, set()).copy():
                    if insn_3 not in dep_graph.get(insn_1, set()):
                        changed_something = True
                        dep_graph[insn_1].add(insn_3)

        if not changed_something:
            break

    for insn_1 in dep_graph:
        for insn_2 in dep_graph.get(insn_1, set()).copy():
            for insn_3 in dep_graph.get(insn_2, set()).copy():
                if insn_3 in dep_graph.get(insn_1, set()):
                    dep_graph[insn_1].remove(insn_3)

    # }}}

    for insn_1 in dep_graph:
        for insn_2 in dep_graph.get(insn_1, set()):
            lines.append("%s -> %s" % (insn_2, insn_1))

    for (insn_1, insn_2), annot in annotation_dep_graph.iteritems():
            lines.append(
                    "%s -> %s  [label=\"%s\", style=dashed]"
                    % (insn_2, insn_1, annot))

    lines.append("subgraph cluster_1 { label=\"initialization\"")
    for dep in code.initialization_dep_on:
        lines.append(dep)
    lines.append("}")

    lines.append("subgraph cluster_2 { label=\"step\"")
    for dep in code.step_dep_on:
        lines.append(dep)
    lines.append("}")

    return "digraph leap_code {\n%s\n}" % (
            "\n".join(lines)
            )


def show_dependency_graph(*args, **kwargs):
    """Show the dependency graph generated by :func:`get_dot_dependency_graph`
    in a browser. Accepts the same arguments as that function.
    """

    dot = get_dot_dependency_graph(*args, **kwargs)

    from tempfile import mkdtemp
    temp_dir = mkdtemp(prefix="tmp_leap_dot")

    dot_file_name = "leap.dot"

    from os.path import join
    with open(join(temp_dir, dot_file_name), "w") as dotf:
        dotf.write(dot)

    svg_file_name = "leap.svg"
    from subprocess import check_call
    check_call(["dot", "-Tsvg", "-o", svg_file_name, dot_file_name],
            cwd=temp_dir)

    full_svg_file_name = join(temp_dir, svg_file_name)
    logger.info("show_dot_dependency_graph: svg written to '%s'"
            % full_svg_file_name)

    from webbrowser import open as browser_open
    browser_open("file://" + full_svg_file_name)

# }}}

# vim: fdm=marker
