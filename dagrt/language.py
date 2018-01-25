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

from pytools import RecordWithoutPickling, memoize_method
from pymbolic.imperative.instruction import (
        ConditionalInstruction as InstructionBase,
        ConditionalAssignment as AssignExpressionBase,
        Nop as NopBase)

from dagrt.utils import get_variables
from contextlib import contextmanager

import logging
import six
import six.moves

logger = logging.getLogger(__name__)

# {{{ instructions

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

Instructions
~~~~~~~~~~~~
.. autoclass:: Instruction

Assignment Instructions
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AssignSolved
.. autoclass:: AssignExpression
.. autoclass:: AssignFunctionCall

State Instructions
^^^^^^^^^^^^^^^^^^

.. autoclass:: YieldState
.. autoclass:: Raise
.. autoclass:: FailStep
.. autoclass:: ExitStep

Code Container
~~~~~~~~~~~~~~

.. autoclass:: ExecutionState
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

def _stringify_instructions(roots, id_to_insn, prefix=""):
        lines = []

        printed_insn_ids = set()

        def print_insn(insn):
            if insn.id in printed_insn_ids:
                return
            printed_insn_ids.add(insn.id)

            for dep_id in insn.depends_on:
                print_insn(id_to_insn[dep_id])

            lines.append(
                    "%s{%s} %s" % (
                        prefix, insn.id, str(insn).replace("\n", "\n        ")))

        for root_insn in roots:
            print_insn(root_insn)

        return lines

# }}}


class Instruction(InstructionBase):
    def get_dependency_mapper(self, include_calls="descend_args"):
        from dagrt.expression import ExtendedDependencyMapper
        return ExtendedDependencyMapper(
            include_subscripts=False,
            include_lookups=False,
            include_calls=include_calls)


class Nop(NopBase):
    exec_method = six.moves.intern("exec_Nop")


# {{{ assignments

class AssignmentBase(Instruction):
    pass


class AssignExpression(Instruction, AssignExpressionBase):
    """
    .. attribute:: loops

        A list of triples *(identifier, start, end)* that the assignment
        should be carried out inside of these loops.
        No ordering of loop iterations is implied.
        The loops will typically be nested outer-to-inner, but a target
        may validly use any order it likes.
    """

    def __init__(self, assignee=None, assignee_subscript=None, expression=None,
            loops=[], **kwargs):

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

        super(AssignExpression, self).__init__(
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
        return (super(AssignExpression, self)
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(
                    loops=[
                        (ident, mapper(start), mapper(end))
                        for ident, start, end in self.loops]))

    def __str__(self):
        result = super(AssignExpression, self).__str__()

        for ident, start, end in self.loops:
            result += " [{ident}={start}..{end}]".format(
                    ident=ident,
                    start=start,
                    end=end)

        return result

    exec_method = "exec_AssignExpression"


class AssignSolved(AssignmentBase):
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
        Instruction.__init__(self, assignees=assignees,
                             solve_variables=solve_variables,
                             expressions=expressions,
                             other_params=other_params,
                             solver_id=solver_id,
                             **kwargs)

    exec_method = six.moves.intern("exec_AssignSolved")

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

        variables = super(AssignSolved, self).get_read_variables()
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

        return (super(AssignSolved, self)
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(
                    assignees=assignees,
                    expressions=mapper(self.expressions)))

    def __str__(self):
        lines = []
        lines.append("AssignSolved")
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
    """This instruction encodes function calls. It should be noted
    that function calls can *also* be encoded as expressions
    involving calls, however the existence of this separate instruction
    is justified by two facts:

    *   Some backends (such as Fortran) choose to separate out function calls
        into individual instructions. This instruction provides a
        natural way to encode them.

        See :class:`leam.vm.codegen.transform.FunctionCallIsolator` for
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

        super(AssignFunctionCall, self).__init__(
                assignees=assignees,
                function_id=function_id,
                parameters=parameters,
                kw_parameters=kw_parameters,
                **kwargs)

    def get_written_variables(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        result = super(AssignFunctionCall, self).get_read_variables()
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

        return (super(AssignFunctionCall, self)
                .map_expressions(mapper, include_lhs=include_lhs)
                .copy(
                    assignees=assignees,
                    function_id=mapped_expr.function.name,
                    parameters=mapped_expr.parameters,
                    kw_parameters=mapped_expr.kw_parameters))

    def __str__(self):
        pars = list(str(p) for p in self.parameters) + [
                "%s=%s" % (name, value)
                for name, value in sorted(self.kw_parameters.items())]

        result = "{assignees} <- {func_id}({pars}){cond}".format(
            assignees=", ".join(self.assignees),
            func_id=self.function_id,
            pars=", ".join(pars),
            cond=self._condition_printing_suffix())

        return result

    exec_method = "exec_AssignFunctionCall"

# }}}


class YieldState(Instruction):
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
                super(YieldState, self).get_read_variables()
                | get_variables(self.expression)
                | get_variables(self.time))

    def map_expressions(self, mapper, include_lhs=True):
        return (super(YieldState, self)
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

    exec_method = six.moves.intern("exec_YieldState")


class Raise(Instruction):
    """
    .. attribute:: error_condition

        A (Python) exception type to be raised.

    .. attribute:: error_message

        The error message to go along with that exception type.
    """

    def __init__(self, error_condition, error_message=None, **kwargs):
        Instruction.__init__(self,
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

    exec_method = six.moves.intern("exec_Raise")


class StateTransition(Instruction):
    """
    .. attribute:: next_state

        The name of the next state to enter
    """

    def __init__(self, next_state, **kwargs):
        Instruction.__init__(self, next_state=next_state, **kwargs)

    def get_written_variables(self):
        return frozenset()

    def __str__(self):
        return "Transition to {state}{cond}".format(state=self.next_state,
            cond=self._condition_printing_suffix())

    exec_method = six.moves.intern("exec_StateTransition")


class FailStep(Instruction):
    """Exits the current step with a failure indication to the controlling
    program. Execution resumes with the next step as normal.
    """

    def get_written_variables(self):
        return frozenset()

    def __str__(self):
        return "FailStep{cond}".format(cond=self._condition_printing_suffix())

    exec_method = six.moves.intern("exec_FailStep")


class ExitStep(Instruction):
    """Exits the current step. Execution resumes with the next step as normal.
    """

    def get_written_variables(self):
        return frozenset()

    def __str__(self):
        return "ExitStep{cond}".format(cond=self._condition_printing_suffix())

    exec_method = six.moves.intern("exec_ExitStep")

# }}}


# {{{ code container

class ExecutionState(RecordWithoutPickling):
    """
    .. attribute:: depends_on

        a list of instruction IDs that need to be accomplished
        for successful execution of one round of this state

    .. attribute:: next_state

        name of the next state after this one, if no other state
        is specified by the user.

    .. attribute:: instructions

        is a list of Instruction instances, in no particular
        order. Only instructions referred to by :attr:`depends_on`
        or the transitive closure of their dependency relations
        will actually be executed.
    """

    def __init__(self, depends_on, next_state, instructions):
        super(ExecutionState, self).__init__(
                depends_on=depends_on,
                next_state=next_state,
                instructions=instructions)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn)
                for insn in self.instructions)


class DAGCode(RecordWithoutPickling):
    """
    .. attribute:: states

        is a map from time integrator state names to :class:`ExecutionState`
        instances

    .. attribute:: initial_state

        the name of the starting state
    """

    @classmethod
    def create_with_steady_state(cls, dep_on, instructions):
        states = {'main': ExecutionState(
            dep_on, next_state='main', instructions=instructions)}
        return cls(states, 'main')

    @classmethod
    def _create_with_init_and_step(cls, initialization_dep_on,
                                  step_dep_on, instructions):
        states = {}
        states['initialization'] = ExecutionState(
                initialization_dep_on,
                next_state='primary',
                instructions=instructions)

        states['primary'] = ExecutionState(
                step_dep_on,
                next_state='primary',
                instructions=instructions)

        return cls(states, 'initialization')

    def __init__(self, states, initial_state):
        assert not isinstance(states, list)
        RecordWithoutPickling.__init__(self,
                                       states=states,
                                       initial_state=initial_state)

    # {{{ identifier wrangling

    def get_insn_id_generator(self):
        from pytools import UniqueNameGenerator
        return UniqueNameGenerator(
                set(insn.id
                    for state in six.itervalues(self.states)
                    for insn in state.instructions))

    def existing_var_names(self):
        result = set()
        for state in six.itervalues(self.states):
            for insn in state.instructions:
                result.update(insn.get_written_variables())
                result.update(insn.get_read_variables())

        return result

    def get_var_name_generator(self):
        from pytools import UniqueNameGenerator
        return UniqueNameGenerator(self.existing_var_names())

    # }}}

    def __str__(self):
        lines = []
        for state_name, state in sorted(six.iteritems(self.states)):
            lines.append("STATE %s" % state_name)

            for root_id in state.depends_on:
                lines.extend(_stringify_instructions(
                    [state.id_to_insn[root_id]],
                    state.id_to_insn, prefix="    "))

            lines.append("    -> (next state) %s" % state.next_state)
            lines.append("")

        return "\n".join(lines)


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
        del self.plan[:]
        self.plan_id_set.clear()
        self.executed_ids.clear()

    def update_plan(self, state, execute_ids):
        """Update the plan with the minimal list of instruction ids to execute
        so that the instruction IDs in execute_ids will be executed before any
        others and such that all their dependencies are satisfied.
        """

        early_plan = []

        id_to_insn = state.id_to_insn

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

    def __call__(self, state, target):
        id_to_insn = state.id_to_insn

        while self.plan:
            insn_id = self.plan.pop(0)
            self.plan_id_set.remove(insn_id)

            self.executed_ids.add(insn_id)

            insn = id_to_insn[insn_id]

            logger.debug("execution trace: [%s] %s" % (
                insn.id, str(insn).replace("\n", " | ")))

            if not target.evaluate_condition(insn):
                continue

            result = getattr(target, insn.exec_method)(insn)
            if result is not None:
                event, new_deps = result
                if event is not None:
                    yield event

                if new_deps is not None:
                    self.update_plan(state, new_deps)

# }}}


# {{{ code building utility

class CodeBuilder(object):
    """
    .. attribute:: instructions

       The set of instructions generated for the state

    .. attribute:: state_dependencies

       A list of instruction names. Starting with these instructions
       as the root dependencies, the state can be executed by following
       the dependency list of each instruction.

    .. automethod:: fence
    .. automethod:: if_
    .. automethod:: else_
    .. automethod:: __call__
    .. automethod:: fresh_var_name
    .. automethod:: fresh_var
    .. automethod:: assign_solved
    .. automethod:: assign_solved_1
    .. automethod:: yield_state
    .. automethod:: fail_step
    .. automethod:: exit_step
    .. automethod:: raise_
    .. automethod:: state_transition
    .. automethod:: __enter__

    """

    class Context(RecordWithoutPickling):
        """
        A context represents a block of instructions being built into the DAG

        .. attribute:: lead_instruction_ids

        .. attribute:: introduced_condition

        .. attribute:: context_instruction_ids

        .. attribute:: used_variables

        .. attribute:: definition_map
        """
        def __init__(self, lead_instruction_ids=[], definition_map={},
                     used_variables=[], condition=True):
            RecordWithoutPickling.__init__(self,
                lead_instruction_ids=frozenset(lead_instruction_ids),
                context_instruction_ids=set(lead_instruction_ids),
                definition_map=dict(definition_map),
                used_variables=set(used_variables),
                condition=condition)

    def __init__(self, label="state"):
        """
        :arg label: The name of the state to generate
        """
        self.label = label
        self._instruction_map = {}
        self._instruction_count = 0
        self._contexts = []
        self._last_popped_context = None
        self._all_var_names = set()
        self._all_generated_var_names = set()

    def fence(self):
        """
        Enter a new logical block of instructions. Force all prior
        instructions to execute before subsequent ones.
        """
        self._contexts[-1] = self._make_new_context(Nop(),
            additional_condition=self._contexts[-1].condition)

    def _get_active_condition(self):
        def is_nontrivial_condition(cond):
            return cond is not True

        conditions = list(filter(is_nontrivial_condition,
                         [context.condition for context in self._contexts]))
        num_conditions = len(conditions)

        # No conditions - trival
        if num_conditions == 0:
            return True

        # Single condition
        if num_conditions == 1:
            return conditions[0]

        # Conjunction of conditions
        from pymbolic.primitives import LogicalAnd
        return LogicalAnd(tuple(conditions))

    @contextmanager
    def if_(self, *condition_arg):
        """Create a new block that is conditionally executed."""
        if len(condition_arg) == 1:
            condition = condition_arg[0]

            from dagrt.expression import parse

            if isinstance(condition, str):
                condition = parse(condition)

        elif len(condition_arg) == 3:
            from pymbolic.primitives import Comparison
            condition = Comparison(*condition_arg)
        else:
            raise ValueError("Unrecognized condition expression")

        # Create an instruction as a lead instruction to assign a logical flag.
        cond_var = self.fresh_var("<cond>")
        cond_assignment = AssignExpression(
                assignee=cond_var.name,
                assignee_subscript=(),
                expression=condition)

        self._contexts.append(
            self._make_new_context(cond_assignment, additional_condition=cond_var))

        yield

        # Pop myself from the stack.
        last_context = self._contexts.pop()
        self._contexts[-1] = self._make_new_context(
            Nop(depends_on=last_context.context_instruction_ids),
            additional_condition=self._contexts[-1].condition)

        self._last_popped_if = last_context

    @contextmanager
    def else_(self):
        """
        Create the "else" portion of a conditionally executed block.
        """
        assert self._last_popped_if

        # Create conditions for the context.
        from pymbolic.primitives import LogicalNot
        self._contexts.append(
            self._make_new_context(Nop(),
                additional_condition=LogicalNot(self._last_popped_if.condition)))

        self._last_popped_if = None

        yield

        # Pop myself from the stack.
        last_context = self._contexts.pop()

        self._contexts[-1] = self._make_new_context(
            Nop(depends_on=last_context.context_instruction_ids),
            additional_condition=self._contexts[-1].condition)

    def _next_instruction_id(self):
        self._instruction_count += 1
        return self.label + "_" + str(self._instruction_count)

    def __call__(self, assignees, expression, loops=[]):
        """Generate code for an assignment.

        *assignees* may be a variable, a subscript (if referring to an
        array), or a tuple of variables. There must be exactly one
        assignee unless *expression* is a function call.
        """

        from dagrt.expression import parse

        def _parse_if_necessary(s):
            if isinstance(s, str):
                return parse(s)
            else:
                return s

        assignees = _parse_if_necessary(assignees)
        if isinstance(assignees, tuple):
            assignees = tuple(
                    _parse_if_necessary(s)
                    for s in assignees)
        else:
            assignees = (assignees,)

        expression = _parse_if_necessary(expression)

        new_loops = []
        for ident, start, stop in loops:
            start = _parse_if_necessary(start)
            stop = _parse_if_necessary(stop)
            new_loops.append((ident, start, stop))

        from pymbolic.primitives import Call, CallWithKwargs, Variable

        if isinstance(expression, (Call, CallWithKwargs)):
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

            self._add_inst_to_context(AssignFunctionCall(
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

            self._add_inst_to_context(AssignExpression(
                    assignee=aname,
                    assignee_subscript=asub,
                    expression=expression,
                    loops=new_loops))

    assign = __call__

    def _add_inst_to_context(self, inst):
        inst_id = self._next_instruction_id()
        context = self._contexts[-1]
        dependencies = set(context.lead_instruction_ids)

        # Verify that assignees are not being places after uses of the
        # assignees in this context.
        for assignee in inst.get_written_variables():
            # Warn about potential ordering of assignments that may
            # be unexpected by the user.
            if assignee in context.used_variables:
                raise ValueError("write after use of " + assignee +
                                 " in the same block")

            if (
                    assignee in context.definition_map

                    # multiple assignments with subscript are OK
                    and not (
                        isinstance(inst, AssignExpression)
                        and inst.assignee_subscript is not None)):
                raise ValueError("multiple assignments to " + assignee)

        # Create the set of dependencies based on the set of used
        # variables.
        for used_variable in inst.get_read_variables():
            if used_variable in context.definition_map:
                dependencies.update(context.definition_map[used_variable])

        for used_variable in inst.get_written_variables():
            # Make second (indexed) writes depend on initialization
            for def_inst_id in context.definition_map.get(used_variable, []):
                def_inst = self._instruction_map[def_inst_id]
                if (
                        not isinstance(def_inst, AssignExpression)
                        or def_inst.assignee_subscript is None):
                    dependencies.add(def_inst_id)

        # Add the condition to the instruction.
        # Update context and global information.
        context.context_instruction_ids.add(inst_id)
        for assignee in inst.get_written_variables():
            context.definition_map.setdefault(assignee, set()).add(inst_id)

        context.used_variables |= inst.get_read_variables()
        self._all_var_names |= inst.get_written_variables()
        self._instruction_map[inst_id] = \
            inst.copy(id=inst_id, depends_on=list(dependencies),
                      condition=self._get_active_condition())
        return inst_id

    def _make_new_context(self, inst, additional_condition=True):
        """
        :param leading_instructions: A list of lead instruction ids
        :conditions: A
        """
        inst_id = self._next_instruction_id()
        context = self._contexts[-1]
        new_context = CodeBuilder.Context(
            lead_instruction_ids=[inst_id],
            used_variables=set(),
            condition=additional_condition)
        self._instruction_map[inst_id] = \
            inst.copy(id=inst_id,
                      depends_on=inst.depends_on | context.context_instruction_ids,
                      condition=self._get_active_condition())
        return new_context

    def fresh_var_name(self, prefix="temp"):
        """Return a variable name that is not guaranteed not to be in
        use and not to be generated in the future."""
        from pytools import generate_unique_names
        for possible_var in generate_unique_names(str(prefix)):
            if possible_var not in self._all_var_names \
                    and possible_var not in self._all_generated_var_names:
                self._all_generated_var_names.add(possible_var)
                return possible_var

    def fresh_var(self, prefix="temp"):
        from pymbolic import var
        return var(self.fresh_var_name(prefix))

    def assign_solved_1(self, assignee, solve_component, expression, guess,
                        solver_id):
        """Special case of AssignSolved when there is 1 component to solve for."""
        self.assign_solved((assignee.name,), (solve_component.name,), (expression,),
                           {"guess": guess}, solver_id)

    def assign_solved(self, assignees, solve_components, expressions,
                      other_params, solver_id):
        self._add_inst_to_context(AssignSolved(assignees, solve_components,
            expressions, other_params, solver_id))

    def yield_state(self, expression, component_id, time, time_id):
        """Yield a value."""

        from dagrt.expression import parse

        if isinstance(expression, str):
            expression = parse(expression)

        self._add_inst_to_context(YieldState(
                expression=expression,
                component_id=component_id,
                time=time,
                time_id=time_id))

    def fail_step(self):
        self.fence()
        self._add_inst_to_context(FailStep())

    def exit_step(self):
        self.fence()
        self._add_inst_to_context(ExitStep())

    def raise_(self, error_condition, error_message=None):
        self.fence()
        self._add_inst_to_context(Raise(error_condition, error_message))

    def state_transition(self, next_state):
        self.fence()
        self._add_inst_to_context(StateTransition(next_state))

    def __enter__(self):
        self._contexts.append(CodeBuilder.Context())
        return self

    def __exit__(self, *ignored):
        self.fence()
        self.state_dependencies = list(self._contexts[-1].lead_instruction_ids)
        self.instructions = set(self._instruction_map.values())

    def __str__(self):
        roots = [
                self._instruction_map[insn_id]
                for ctx in self._contexts
                for insn_id in ctx.context_instruction_ids]

        return "\n".join(_stringify_instructions(roots, self._instruction_map))

    def as_execution_state(self, next_state):
        """
        :arg cb: A :class:`CodeBuilder` instance
        :arg next_state: The name of the default next state
        """
        return ExecutionState(
                depends_on=self.state_dependencies, next_state=next_state,
                instructions=self.instructions)

# }}}


# {{{ graphviz / dot export

def get_dot_dependency_graph(code, use_insn_ids=False):
    """Return a string in the `dot <http://graphviz.org/>`_ language depicting
    dependencies among kernel instructions.
    """

    from pymbolic.imperative.utils import get_dot_dependency_graph

    def addtional_lines_hook():
        for i, (name, state) in enumerate(six.iteritems(code.states)):
            yield "subgraph cluster_%d { label=\"%s\"" % (i, name)
            for dep in state.depends_on:
                yield dep
            yield "}"

    instructions = [
            insn if use_insn_ids else insn.copy(id=insn.id)
            for state_name, state in six.iteritems(code.states)
            for insn in state.instructions]
    return get_dot_dependency_graph(
            instructions, use_insn_ids=use_insn_ids,
            addtional_lines_hook=addtional_lines_hook)


def show_dependency_graph(*args, **kwargs):
    """Show the dependency graph generated by :func:`get_dot_dependency_graph`
    in a browser. Accepts the same arguments as that function.
    """
    from pymbolic.imperative.utils import show_dot
    show_dot(get_dot_dependency_graph(*args, **kwargs))

# }}}

# vim: fdm=marker
