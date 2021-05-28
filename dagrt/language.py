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

from pytools import RecordWithoutPickling, memoize_method, natsorted
from pymbolic.imperative.statement import (
        ConditionalStatement as StatementBase,
        ConditionalAssignment as AssignBase,
        Nop as NopBase)

from dagrt.utils import get_variables
from contextlib import contextmanager

import logging

from sys import intern


logger = logging.getLogger(__name__)

# {{{ statements

__doc__ = """
Identifier conventions
======================

Identifiers whose names start with the pattern <letters> are special.  The
following special variable names are supported:

Internal names
~~~~~~~~~~~~~~

``<p>NAME``
    This variable contains persistent state.
    (that survives from one step to the next)

``<cond>NAME``
    This variable is used by a conditional. May not be re-defined.

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

User-controlled values
~~~~~~~~~~~~~~~~~~~~~~

``<state>NAME``
    State identifier under user (=scheme writer) control

    This variable contains persistent state.
    (that survives from one step to the next)

``<ret_time_id>COMPONENT_ID``

``<ret_time>COMPONENT_ID``

``<ret_state>COMPONENT_ID``

    For targets that are incapable of returning state mid-step, these variables
    are used to store computed state.

See :mod:`dagrt.function_registry` for interpretation of function names. The
function namespace and the variable namespace are distinct. No user-defined
identifiers should start with `dagrt_`.

Statements
~~~~~~~~~~
.. autoclass:: Statement

Assignment Statements
^^^^^^^^^^^^^^^^^^^^^

These statements perform updates to the execution state, i.e. the variables.

.. autoclass:: AssignImplicit
.. autoclass:: Assign
.. autoclass:: AssignFunctionCall

Control Statements
^^^^^^^^^^^^^^^^^^

These statements affect the execution of a phase, or cause a phase to interact
with user code.

.. autoclass:: YieldState
.. autoclass:: Raise
.. autoclass:: FailStep
.. autoclass:: SwitchPhase

Miscellaneous Statements
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Nop

Code Container
~~~~~~~~~~~~~~

.. autoclass:: ExecutionPhase
.. autoclass:: DAGCode

Visualization
~~~~~~~~~~~~~

.. autofunction:: get_dot_dependency_graph
.. autofunction:: show_dependency_graph

Code Creation
~~~~~~~~~~~~~

.. autoclass :: CodeBuilder

=======

"""


# {{{ utilities

def _stringify_statements(roots, id_to_stmt, prefix=""):
    lines = []

    printed_stmt_ids = set()

    def print_stmt(stmt):
        if stmt.id in printed_stmt_ids:
            return
        printed_stmt_ids.add(stmt.id)

        for dep_id in natsorted(stmt.depends_on):
            print_stmt(id_to_stmt[dep_id])

        lines.append(
                "{}{{{}}} {}".format(
                    prefix, stmt.id, str(stmt).replace("\n", "\n        ")))

    for root_stmt in roots:
        print_stmt(root_stmt)

    return lines

# }}}


class Statement(StatementBase):
    def get_dependency_mapper(self, include_calls="descend_args"):
        from dagrt.expression import ExtendedDependencyMapper
        return ExtendedDependencyMapper(
            include_subscripts=False,
            include_lookups=False,
            include_calls=include_calls)


class Nop(NopBase):
    exec_method = intern("exec_Nop")


# {{{ assignments

class AssignmentBase(Statement):
    pass


class Assign(Statement, AssignBase):
    """
    .. attribute:: loops

        A list of triples *(identifier, start, end)* that the assignment
        should be carried out inside of these loops.
        No ordering of loop iterations is implied.
        The loops will typically be nested outer-to-inner, but a target
        may validly use any order it likes.
    """

    def __init__(self, assignee=None, assignee_subscript=None, expression=None,
            loops=None, **kwargs):
        if loops is None:
            loops = []

        if "lhs" not in kwargs:
            if assignee is None:
                raise TypeError("assignee is a required argument")
            if assignee_subscript is None:
                raise TypeError("assignee is a required argument")

            from pymbolic import var
            lhs = var(assignee)

            if assignee_subscript:
                if not isinstance(assignee_subscript, tuple):
                    raise TypeError("assignee_subscript must be a tuple")

                lhs = lhs[assignee_subscript]
                if len(assignee_subscript) > 1:
                    raise ValueError(
                            "assignee subscript may have a length of at most one")
        else:
            lhs = kwargs.pop("lhs")

        if "rhs" not in kwargs:
            if expression is None:
                raise TypeError("assignee is a required argument")
            rhs = expression
        else:
            rhs = kwargs.pop("rhs")

        super().__init__(
                lhs=lhs,
                rhs=rhs,
                loops=loops,
                **kwargs)

    @property
    def assignee(self):
        from pymbolic.primitives import Variable, Subscript
        if isinstance(self.lhs, Variable):
            return self.lhs.name
        elif isinstance(self.lhs, Subscript):
            assert isinstance(self.lhs.aggregate, Variable)
            return self.lhs.aggregate.name
        else:
            raise TypeError("unexpected type of LHS")

    @property
    def assignee_subscript(self):
        from pymbolic.primitives import Variable, Subscript
        if isinstance(self.lhs, Variable):
            return ()
        elif isinstance(self.lhs, Subscript):
            return self.lhs.index
        else:
            raise TypeError("unexpected type of LHS")

    @property
    def expression(self):
        return self.rhs

    def map_expressions(self, mapper, include_lhs=True):
        return (super()
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(
                    loops=[
                        (ident, mapper(start), mapper(end))
                        for ident, start, end in self.loops]))

    def __str__(self):
        result = super().__str__()

        for ident, start, end in self.loops:
            result += " [{ident}={start}..{end}]".format(
                    ident=ident,
                    start=start,
                    end=end)

        return result

    exec_method = "exec_Assign"


class AssignImplicit(AssignmentBase):
    """
    .. attribute:: assignees

        A tuple of strings. The names of the variables to assign with the
        results of solving for `solve_variables`

    .. attribute:: solve_variables

        A tuple of strings, the names of the variables being solved for

    .. attribute:: expressions

        A tuple of expressions, which represent the left hand sides of a
        system of (possibly nonlinear) equations. The solver will attempt
        to find a simultaneous root of the system.

    .. attribute:: other_params

        A dictionary used to pass extra arguments to the solver, for instance
        a starting guess

    .. attribute:: solver_id

        An identifier for the solver that is to be used to solve the
        system. This identifier is intended to match information about
        solvers which is supplied by the user.

    """

    def __init__(self, assignees, solve_variables, expressions, other_params,
                 solver_id, **kwargs):
        Statement.__init__(self, assignees=assignees,
                             solve_variables=solve_variables,
                             expressions=expressions,
                             other_params=other_params,
                             solver_id=solver_id,
                             **kwargs)

    exec_method = intern("exec_AssignImplicit")

    def get_written_variables(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        # Variables can be read by:
        #  1. expressions (except for those in solve_variables)
        #  2. values in other_params
        #  3. condition
        from itertools import chain

        def flatten(iter_arg):
            return chain(*list(iter_arg))

        variables = super().get_read_variables()
        variables |= set(flatten(get_variables(expr) for expr in self.expressions))
        variables -= set(self.solve_variables)
        variables |= set(flatten(get_variables(expr) for expr
                                 in self.other_params.values()))
        return variables

    def map_expressions(self, mapper, include_lhs=True):
        from pymbolic.primitives import Variable

        if include_lhs:
            lhss = tuple(mapper(Variable(assignee)) for assignee in self.assignees)
            assert all(isinstance(lhs, Variable) for lhs in lhss)
            assignees = tuple(lhs.name for lhs in lhss)
        else:
            assignees = self.assignees

        return (super()
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(
                    assignees=assignees,
                    expressions=mapper(self.expressions)))

    def __str__(self):
        lines = []
        lines.append("AssignImplicit")
        lines.append("solver_id = " + str(self.solver_id))
        for assignee_index, assignee in enumerate(self.assignees):
            lines.append(assignee + " <- " + self.solve_variables[assignee_index])
        lines.append("where")
        for expression in self.expressions:
            lines.append("  " + str(expression) + " = 0")
        if self.other_params:
            lines.append("with parameters")
            for param_name, param_value in self.other_params.items():
                lines.append(param_name + ": " + str(param_value))
        lines.append(self._condition_printing_suffix())
        return "\n".join(lines)


class AssignFunctionCall(AssignmentBase):
    """This statement encodes function calls. It should be noted
    that function calls can *also* be encoded as expressions
    involving calls, however the existence of this separate statement
    is justified by two facts:

    *   Some backends (such as Fortran) choose to separate out function calls
        into individual statements. This statement provides a
        natural way to encode them.

        See :class:`dagrt.codegen.transform.isolate_function_calls` for
        a transform that accomplishes this.

    *   Calling functions with multiple return values is not supported
        as part of dagrt's language.

    .. attribute:: assignees

        A tuple of variables to be passed assigned results from
        calling :attr:`function_id`.

    .. attribute:: function_id
    .. attribute:: parameters

        A list of expressions to be passed as positional arguments.

    .. attribute:: kw_parameters

        A dictionary mapping names to expressions to be passed as keyword
        arguments.
    """

    def __init__(self, assignees, function_id, parameters, kw_parameters=None,
            **kwargs):
        if kw_parameters is None:
            kw_parameters = {}

        super().__init__(
                assignees=assignees,
                function_id=function_id,
                parameters=parameters,
                kw_parameters=kw_parameters,
                **kwargs)

    def get_written_variables(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        result = super().get_read_variables()
        for par in self.parameters:
            result |= get_variables(par)
        for par in self.kw_parameters.values():
            result |= get_variables(par)

        return result

    def as_expression(self):
        from pymbolic.primitives import CallWithKwargs, Variable
        return CallWithKwargs(
                Variable(self.function_id),
                parameters=self.parameters,
                kw_parameters=self.kw_parameters)

    def map_expressions(self, mapper, include_lhs=True):
        from pymbolic.primitives import CallWithKwargs, Variable
        mapped_expr = mapper(self.as_expression())
        assert isinstance(mapped_expr, CallWithKwargs)

        if include_lhs:
            lhss = tuple(mapper(Variable(assignee)) for assignee in self.assignees)
            assert all(isinstance(lhs, Variable) for lhs in lhss)
            assignees = tuple(lhs.name for lhs in lhss)
        else:
            assignees = self.assignees

        return (super()
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(
                    assignees=assignees,
                    function_id=mapped_expr.function.name,
                    parameters=mapped_expr.parameters,
                    kw_parameters=mapped_expr.kw_parameters))

    def __str__(self):
        pars = list(str(p) for p in self.parameters) + [
                f"{name}={value}"
                for name, value in sorted(self.kw_parameters.items())]

        result = "{assignees} <- {func_id}({pars}){cond}".format(
            assignees=", ".join(self.assignees),
            func_id=self.function_id,
            pars=", ".join(pars),
            cond=self._condition_printing_suffix())

        return result

    exec_method = "exec_AssignFunctionCall"

# }}}


class YieldState(Statement):
    """
    .. attribute:: time_id
    .. attribute:: time

        A :mod:`pymbolic` expression representing the time at which the
        state returned is valid.

    .. attribute:: component_id
    .. attribute:: expression
    """

    def get_written_variables(self):
        return frozenset()

    def get_read_variables(self):
        return (
                super().get_read_variables()
                | get_variables(self.expression)
                | get_variables(self.time))

    def map_expressions(self, mapper, include_lhs=True):
        return (super()
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(expression=mapper(self.expression),
                      time=mapper(self.time)))

    def __str__(self):
        return ("Ret {expr} at {time_id} with t={time} as {component_id}{cond}"
                .format(
                    expr=self.expression,
                    time_id=self.time_id,
                    time=self.time,
                    component_id=self.component_id,
                    cond=self._condition_printing_suffix()))

    exec_method = intern("exec_YieldState")


class Raise(Statement):
    """
    .. attribute:: error_condition

        A (Python) exception type to be raised.

    .. attribute:: error_message

        The error message to go along with that exception type.
    """

    def __init__(self, error_condition, error_message=None, **kwargs):
        Statement.__init__(self,
                error_condition=error_condition,
                error_message=error_message,
                **kwargs)

    def get_written_variables(self):
        return frozenset()

    def __str__(self):
        error = self.error_condition.__name__
        if self.error_message:
            error += "(" + repr(self.error_message) + ")"

        return "Raise {error}{cond}".format(error=error,
                                            cond=self._condition_printing_suffix())

    exec_method = intern("exec_Raise")


class SwitchPhase(Statement):
    """
    .. attribute:: next_phase

        The name of the next state to enter
    """

    def __init__(self, next_phase, **kwargs):
        Statement.__init__(self, next_phase=next_phase, **kwargs)

    def get_written_variables(self):
        return frozenset()

    def __str__(self):
        return "Transition to {state}{cond}".format(state=self.next_phase,
            cond=self._condition_printing_suffix())

    exec_method = intern("exec_SwitchPhase")


class FailStep(Statement):
    """Exits the current step with a failure indication to the controlling
    program. Execution resumes with the next step as normal.
    """

    def get_written_variables(self):
        return frozenset()

    def __str__(self):
        return f"FailStep{self._condition_printing_suffix()}"

    exec_method = intern("exec_FailStep")

# }}}


# {{{ code container

class ExecutionPhase(RecordWithoutPickling):
    """
    .. attribute:: name

        name of this phase

    .. attribute:: depends_on

        a list of statement IDs that need to be executed
        to complete a successful execution of one round of
        this phase, excluding their dependencies.

    .. attribute:: next_phase

        name of the next state after this one, if no other state
        is specified by the user.

    .. attribute:: statements

        is a list of statement instances in no particular
        order.
    """

    def __init__(self, name, next_phase, statements):
        super().__init__(
                name=name,
                next_phase=next_phase,
                statements=statements)

    @property
    @memoize_method
    def depends_on(self):
        # Get statement IDs which no other statement depends on.
        result = {stmt.id for stmt in self.statements}
        for stmt in self.statements:
            result -= set(stmt.depends_on)
        return result

    @property
    @memoize_method
    def id_to_stmt(self):
        return {stmt.id: stmt
                for stmt in self.statements}


class DAGCode(RecordWithoutPickling):
    """
    .. attribute:: phases

        is a map from time integrator phase names to :class:`ExecutionPhase`
        instances

    .. attribute:: initial_phase

        the name of the starting phase
    """

    @classmethod
    def from_phases_list(cls, phases, initial_phase):
        name_to_phase = dict()
        for phase in phases:
            if phase.name in name_to_phase:
                raise ValueError("duplicate phase name '%s'" % phase.name)
            name_to_phase[phase.name] = phase
        return cls(name_to_phase, initial_phase)

    def __init__(self, phases, initial_phase):
        assert not isinstance(phases, list)
        RecordWithoutPickling.__init__(self,
                                       phases=phases,
                                       initial_phase=initial_phase)

    # {{{ identifier wrangling

    def get_stmt_id_generator(self):
        from pytools import UniqueNameGenerator
        return UniqueNameGenerator(
                {stmt.id
                    for phase in self.phases.values()
                    for stmt in phase.statements})

    def existing_var_names(self):
        result = set()
        for state in self.phases.values():
            for stmt in state.statements:
                result.update(stmt.get_written_variables())
                result.update(stmt.get_read_variables())

        return result

    def get_var_name_generator(self):
        from pytools import UniqueNameGenerator
        return UniqueNameGenerator(self.existing_var_names())

    # }}}

    def __str__(self):
        lines = []
        for phase_name, phase in sorted(self.phases.items()):
            phase_title = 'PHASE "%s"' % phase_name
            if phase_name == self.initial_phase:
                phase_title += " (initial_phase)"
            lines.append(phase_title)

            for root_id in natsorted(phase.depends_on):
                lines.extend(_stringify_statements(
                    [phase.id_to_stmt[root_id]],
                    phase.id_to_stmt, prefix="    "))

            lines.append('    -> (next phase) "%s"' % phase.next_phase)
            lines.append("")

        return "\n".join(lines)


# }}}


# {{{ interpreter foundation

class ExecutionController:

    def __init__(self, code):
        self.code = code

        self.plan = []
        self.executed_ids = set()
        self.plan_id_set = set()

    def reset(self):
        logger.debug("execution reset")
        del self.plan[:]
        self.plan_id_set.clear()
        self.executed_ids.clear()

    def update_plan(self, phase, execute_ids):
        """Update the plan with the minimal list of statement ids to execute
        so that the statement IDs in execute_ids will be executed before any
        others and such that all their dependencies are satisfied.
        """

        early_plan = []

        id_to_stmt = phase.id_to_stmt

        def add_with_deps(stmt):
            stmt_id = stmt.id
            if stmt_id in self.executed_ids:
                # Already done, no need to think more.
                return

            if stmt_id in self.plan_id_set:
                # Already in plan, no need to think more.
                return

            if stmt_id in early_plan:
                return

            for dep_id in stmt.depends_on:
                add_with_deps(id_to_stmt[dep_id])

            assert stmt_id not in self.plan_id_set

            early_plan.append(stmt_id)

        for stmt_id in execute_ids:
            add_with_deps(id_to_stmt[stmt_id])

        self.plan = early_plan + self.plan
        self.plan_id_set.update(early_plan)

    def __call__(self, phase, target):
        id_to_stmt = phase.id_to_stmt

        while self.plan:
            stmt_id = self.plan.pop(0)
            self.plan_id_set.remove(stmt_id)

            self.executed_ids.add(stmt_id)

            stmt = id_to_stmt[stmt_id]

            logger.debug("execution trace: [{}] {}".format(
                stmt.id, str(stmt).replace("\n", " | ")))

            if not target.evaluate_condition(stmt):
                continue

            result = getattr(target, stmt.exec_method)(stmt)
            if result is not None:
                event, new_deps = result
                if event is not None:
                    yield event

                if new_deps is not None:
                    self.update_plan(phase, new_deps)

# }}}


# {{{ code building utility

class CodeBuilder:
    """
    .. attribute:: name

       The name of the phase being generated

    .. attribute:: statements

       The set of statements generated for the phase

    **Language support:**

    .. automethod:: assign
    .. automethod:: assign_implicit
    .. automethod:: yield_state
    .. automethod:: fail_step
    .. automethod:: raise_
    .. automethod:: switch_phase

    **Control flow:**

    .. automethod:: if_
    .. automethod:: else_

    **Context manager:**

    .. automethod:: __enter__
    .. automethod:: __exit__

    **Graph generation:**

    .. automethod:: as_execution_phase

    **Convenience functions:**

    .. method:: __call__

        Alias for :func:`CodeBuilder.assign`.

    .. automethod:: assign_implicit_1
    .. automethod:: fresh_var_name
    .. automethod:: fresh_var
    .. automethod:: restart_step
    """

    # This is a dummy variable name representing the "system state", which is
    # used to track the dependencies for statements having globally visible
    # side effects.
    _EXECUTION_STATE = "<exec>"

    def __init__(self, name):
        """
        :arg name: The name of the phase to generate
        """
        self.name = name
        self.statements = []

        # Maps variables to the sequentially last statement to write them
        self._writer_map = {}
        # Maps variables to the set of statements that read them between the
        # last time the var was written and the current statement
        self._reader_map = {}
        # Stack of conditional expressions used to implement nested ifs
        self._conditional_expression_stack = []
        # Used to implement if/else
        self._last_if_block_conditional_expression = None
        # Set of seen variables
        self._seen_var_names = {self._EXECUTION_STATE}

    def _add_statement(self, stmt):
        stmt_id = self.next_statement_id()

        read_variables = set(stmt.get_read_variables())
        written_variables = set(stmt.get_written_variables())

        # Add the global execution state as an implicitly read variable.
        read_variables.add(self._EXECUTION_STATE)

        # Build the condition attribute.
        if not self._conditional_expression_stack:
            condition = True
        elif len(self._conditional_expression_stack) == 1:
            condition = self._conditional_expression_stack[0]
        else:
            from pymbolic.primitives import LogicalAnd
            condition = LogicalAnd(tuple(self._conditional_expression_stack))

        from dagrt.utils import get_variables
        read_variables |= get_variables(condition)

        is_non_assignment = (
                not isinstance(
                    stmt, (Assign, AssignImplicit, AssignFunctionCall)))

        # We regard all non-assignments as having potential external side
        # effects (i.e., writing to EXECUTION_STATE).  To keep the global
        # variables in a well-defined state, ensure that all updates to global
        # variables have happened before a non-assignment.
        if is_non_assignment:
            from dagrt.utils import is_state_variable
            read_variables |= {
                    var for var in self._seen_var_names if
                    is_state_variable(var)}
            written_variables.add(self._EXECUTION_STATE)

        depends_on = set()

        # Ensure this statement happens after the last write of all the
        # variables it reads or writes.
        for var in read_variables | written_variables:
            writer = self._writer_map.get(var, None)
            if writer is not None:
                depends_on.add(writer)

        # Ensure this statement happens after the last read(s) of the variables
        # it writes to.
        for var in written_variables:
            readers = self._reader_map.get(var, set())
            depends_on |= readers
            # Keep the graph sparse by clearing the readers set.
            readers.clear()

        for var in written_variables:
            self._writer_map[var] = stmt_id

        for var in read_variables:
            # reader_map should ignore reads that happen before writes, so
            # ignore if this statement also reads *var*.
            if var in written_variables:
                continue
            self._reader_map.setdefault(var, set()).add(stmt_id)

        stmt = stmt.copy(
                id=stmt_id,
                condition=condition,
                depends_on=frozenset(depends_on))
        self.statements.append(stmt)
        self._seen_var_names |= read_variables | written_variables

    def next_statement_id(self):
        return "%s_%d" % (self.name, len(self.statements))

    @memoize_method
    def _var_name_generator(self, prefix):
        from pytools import generate_unique_names
        return generate_unique_names(prefix)

    def fresh_var_name(self, prefix="temp"):
        """Return a variable name that is not in use also and won't be returned in the
        future, regardless of use.
        """
        for var_name in self._var_name_generator(prefix):
            if var_name not in self._seen_var_names:
                self._seen_var_names.add(var_name)
                return var_name

    def fresh_var(self, prefix="temp"):
        from pymbolic import var
        return var(self.fresh_var_name(prefix))

    @contextmanager
    def if_(self, *condition_arg):
        """Create a new block that is conditionally executed."""
        from dagrt.expression import parse

        if len(condition_arg) == 1:
            condition = condition_arg[0]

            if isinstance(condition, str):
                condition = parse(condition)

        elif len(condition_arg) == 3:
            lhs, cond, rhs = condition_arg

            if isinstance(lhs, str):
                lhs = parse(lhs)
            if isinstance(rhs, str):
                rhs = parse(lhs)

            from pymbolic.primitives import Comparison
            condition = Comparison(lhs, cond, rhs)
        else:
            raise ValueError("Unrecognized condition expression")

        # Create an statement as a lead statement to assign a logical flag.
        cond_var = self.fresh_var("<cond>")
        cond_assignment = Assign(
                assignee=cond_var.name,
                assignee_subscript=(),
                expression=condition)

        self._add_statement(cond_assignment)

        self._conditional_expression_stack.append(cond_var)
        yield
        self._conditional_expression_stack.pop()
        self._last_if_block_conditional_expression = cond_var

    @contextmanager
    def else_(self):
        """
        Create the "else" portion of a conditionally executed block.
        """
        assert self._last_if_block_conditional_expression is not None

        # Create conditions for the context.
        from pymbolic.primitives import LogicalNot
        self._conditional_expression_stack.append(
                LogicalNot(self._last_if_block_conditional_expression))
        yield
        self._conditional_expression_stack.pop()
        self._last_if_block_conditional_expression = None

    def assign(self, assignees, expression, loops=None):
        """Generate code for an assignment.

        *assignees* may be a variable, a subscript (if referring to an
        array), or a tuple of variables. There must be exactly one
        assignee unless *expression* is a function call.

        *loops* is a list of tuples of the form
        *(identifier, start_index, stop_index)*.
        """
        if loops is None:
            loops = []

        from dagrt.expression import parse

        def parse_if_necessary(s):
            if isinstance(s, str):
                return parse(s)
            else:
                return s

        assignees = parse_if_necessary(assignees)
        if isinstance(assignees, tuple):
            assignees = tuple(
                    parse_if_necessary(s)
                    for s in assignees)
        else:
            assignees = (assignees,)

        expression = parse_if_necessary(expression)

        new_loops = []
        for ident, start, stop in loops:
            start = parse_if_necessary(start)
            stop = parse_if_necessary(stop)
            new_loops.append((ident, start, stop))

        from pymbolic.primitives import Call, CallWithKwargs, Variable

        if isinstance(expression, (Call, CallWithKwargs)) and not loops:
            assignee_names = []
            for a in assignees:
                if not isinstance(a, Variable):
                    raise ValueError("all assignees (left-hand sides)"
                            "must be plain variables--'%s' is not" % a)

                assignee_names.append(a.name)

            if isinstance(expression, CallWithKwargs):
                kw_parameters = expression.kw_parameters
            else:
                kw_parameters = {}

            self._add_statement(AssignFunctionCall(
                    assignees=assignee_names,
                    function_id=expression.function.name,
                    parameters=expression.parameters,
                    kw_parameters=kw_parameters))

        else:
            if len(assignees) != 1:
                raise ValueError(
                        "exactly one assignee (left-hand side) expected--"
                        "%d found" % len(assignees))
            assignee, = assignees

            from pymbolic.primitives import Variable, Subscript
            if isinstance(assignee, Variable):
                aname = assignee.name
                asub = ()

            elif isinstance(assignee, Subscript):
                aname = assignee.aggregate.name
                asub = assignee.index
                if not isinstance(asub, tuple):
                    asub = (asub,)

            else:
                raise ValueError("assignee (left-hand side) must be either a "
                        "variable or a subscribted variable, not '%s'"
                        % type(assignee))

            self._add_statement(Assign(
                    assignee=aname,
                    assignee_subscript=asub,
                    expression=expression,
                    loops=new_loops))

    __call__ = assign

    def assign_implicit_1(self, assignee, solve_component, expression, guess,
                        solver_id=None):
        """Special case of AssignImplicit when there is 1 component to solve for."""
        self.assign_implicit(
                (assignee.name,), (solve_component.name,), (expression,),
                {"guess": guess}, solver_id)

    def assign_implicit(self, assignees, solve_components, expressions,
                      other_params, solver_id):
        self._add_statement(AssignImplicit(
                assignees, solve_components,
                expressions, other_params, solver_id))

    def yield_state(self, expression, component_id, time, time_id):
        """Yield a value."""

        from dagrt.expression import parse

        if isinstance(expression, str):
            expression = parse(expression)

        self._add_statement(YieldState(
                expression=expression,
                component_id=component_id,
                time=time,
                time_id=time_id))

    def fail_step(self):
        self._add_statement(FailStep())

    def raise_(self, error_condition, error_message=None):
        self._add_statement(Raise(error_condition, error_message))

    def restart_step(self):
        """Equivalent to *switch_phase(self.name)*."""
        self.switch_phase(self.name)

    def switch_phase(self, next_phase):
        self._add_statement(SwitchPhase(next_phase))

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        pass

    def as_execution_phase(self, next_phase):
        """Return the generated graph as an :class:`ExecutionPhase`.

        :arg next_phase: The name of the default next phase
        """
        return ExecutionPhase(
                name=self.name,
                next_phase=next_phase,
                statements=frozenset(self.statements))

# }}}


# {{{ graphviz / dot export

def get_dot_dependency_graph(code, use_stmt_ids=False):
    """Return a string in the `dot <http://graphviz.org/>`_ language depicting
    dependencies among kernel statements.
    """

    from pymbolic.imperative.utils import get_dot_dependency_graph

    def additional_lines_hook():
        for i, (name, phase) in enumerate(code.phases.items()):
            yield 'subgraph cluster_%d { label="%s"' % (i, name)
            yield from natsorted(phase.depends_on)
            yield "}"

    statements = [
            stmt if use_stmt_ids else stmt.copy(id=stmt.id)
            for phase_name, phase in code.phases.items()
            for stmt in natsorted(phase.statements, key=lambda stmt: stmt.id)]

    return get_dot_dependency_graph(
            statements, use_stmt_ids=use_stmt_ids,
            additional_lines_hook=additional_lines_hook)


def show_dependency_graph(*args, **kwargs):
    """Show the dependency graph generated by :func:`get_dot_dependency_graph`
    in a browser. Accepts the same arguments as that function.
    """
    from pymbolic.imperative.utils import show_dot
    show_dot(get_dot_dependency_graph(*args, **kwargs))

# }}}

# vim: fdm=marker
