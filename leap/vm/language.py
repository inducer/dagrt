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
from leap.vm.utils import get_variables
from contextlib import contextmanager

import logging
import six
import six.moves

logger = logging.getLogger(__name__)

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
    State identifier under user (=scheme writer) control

    This variable contains persistent state.
    (that survives from one step to the next)

``<ret_time_id>COMPONENT_ID``
``<ret_time>COMPONENT_ID``
``<ret_state>COMPONENT_ID``
    For targets that are incapable of returning state mid-step, these variables
    are used to store computed state.

The latter two serve to separate the name space used by the method from that
under the control of the user.

See :module:`leap.vm.function_registry` for interpretation of function names.
The function namespace and the variable namespace are distinct. No user-defined
identifiers should start with `leap_`.

Instructions
~~~~~~~~~~~~
.. autoclass:: Instruction

Assignment Instructions
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AssignSolved

.. autoclass:: AssignExpression

State Instructions
^^^^^^^^^^^^^^^^^^

.. autoclass:: YieldState

.. autoclass:: Raise

.. autoclass:: FailStep

.. autoclass:: If

Code Container
~~~~~~~~~~~~~~

.. autoclass:: TimeIntegratorState
.. autoclass:: TimeIntegratorCode

Visualization
~~~~~~~~~~~~~

.. autofunction:: get_dot_dependency_graph

.. autofunction:: show_dependency_graph

"""


class Instruction(RecordWithoutPickling):
    """
    .. attribute:: condition

       The instruction condition as a :mod:`pymbolic` expression (`True` if the
       instruction is unconditionally executed)

    .. attribute:: depends_on

        A :class:`frozenset` of instruction ids that are reuqired to be
        executed within this execution context before this instruction can be
        executed.

    .. attribute:: id

        A string, a unique identifier for this instruction.

    .. automethod:: get_assignees
    .. automethod:: get_read_variables

    """

    def __init__(self, **kwargs):
        id = kwargs.pop("id", None)
        if id is not None:
            id = six.moves.intern(id)
        condition = kwargs.pop("condition", True)
        depends_on = frozenset(kwargs.pop("depends_on", []))
        RecordWithoutPickling.__init__(self,
                                       id=id,
                                       condition=condition,
                                       depends_on=depends_on,
                                       **kwargs)

    def _condition_printing_suffix(self):
        if self.condition is True:
            return ""
        return " if " + str(self.condition)

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

    def map_expressions(self, mapper):
        """Returns a new copy of *self* with all expressions
        replaced by ``mapepr(expr)`` for every
        :class:`pymbolic.primitives.Expression`
        contained in *self*.
        """
        raise NotImplementedError()


class Nop(Instruction):

    def get_assignees(self):
        return frozenset([])

    get_read_variables = get_assignees

    def map_expressions(self, mapper):
        return self.copy(condition=mapper(self.condition))

    def __str__(self):
        return 'nop'

    exec_method = six.moves.intern("exec_Nop")


# {{{ assignments

class AssignmentBase(Instruction):
    pass


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

    def get_assignees(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        # Variables can be read by:
        #  1. expressions (except for those in solve_variables)
        #  2. values in other_params
        #  3. condition
        from itertools import chain
        flatten = lambda iter_arg: chain(*list(iter_arg))
        variables = frozenset()
        variables |= set(flatten(get_variables(expr) for expr in self.expressions))
        variables -= set(self.solve_variables)
        variables |= set(flatten(get_variables(expr) for expr
                                 in self.other_params.values()))
        variables |= get_variables(self.condition)
        return variables

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


class AssignExpression(AssignmentBase):
    """
    .. attribute:: assignee
    .. attribute:: assignee_subscript

        The subscript in :attr:`assignee` which is being assigned.
        A tuple, which may be empty, to indicate 'no subscript'.

    .. attribute:: expression
    .. attribute:: loops

        A list of triples *(identifier, start, end)* that the assignment
        should be carried out inside of these loops.
        No ordering of loop iterations is implied.
        The loops will typically be nested outer-to-inner, but a target
        may validly use any order it likes.

    """
    def __init__(self, assignee, assignee_subscript, expression, loops=[], **kwargs):
        assert isinstance(assignee_subscript, tuple)

        if len(assignee_subscript) > 1:
            raise ValueError("assignee subscript may have a length of at most one")

        Instruction.__init__(self,
                assignee=assignee,
                assignee_subscript=assignee_subscript,
                expression=expression,
                loops=loops,
                **kwargs)

    def get_assignees(self):
        return frozenset([self.assignee])

    def get_read_variables(self):
        result = (get_variables(self.expression)
                | get_variables(self.condition)
                | get_variables(self.assignee_subscript))

        for ident, start, end in self.loops:
            result = result | (get_variables(start) | get_variables(end))

        return result

    def map_expressions(self, mapper):
        return self.copy(
                expression=mapper(self.expression),
                condition=mapper(self.condition),
                loops=[
                    (ident, mapper(start), mapper(end))
                    for ident, start, end in self.loops])

    def __str__(self):
        assignee = self.assignee
        if self.assignee_subscript:
            assignee += "[%s]" % ", ".join(str(ax) for ax in self.assignee_subscript)

        result = "{assignee} <- {expr}{cond}".format(
            assignee=assignee,
            expr=str(self.expression),
            cond=self._condition_printing_suffix())

        for ident, start, end in self.loops:
            result += " [{ident}={start}..{end}]".format(
                    ident=ident,
                    start=start,
                    end=end)

        return result

    exec_method = "exec_AssignExpression"

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

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return get_variables(self.expression) | get_variables(self.time)

    def map_expressions(self, mapper):
        return self.copy(
                expression=mapper(self.expression),
                condition=mapper(self.condition))

    def __str__(self):
        return "Ret {expr} at {time_id} as {component_id}{cond}".format(
                    expr=self.expression,
                    time_id=self.time_id,
                    component_id=self.component_id,
                    cond=self._condition_printing_suffix())

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

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return get_variables(self.condition)

    def map_expressions(self, mapper):
        return self.copy(condition=mapper(self.condition))

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

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return get_variables(self.condition)

    def map_expressions(self, mapper):
        return self.copy(condition=mapper(self.condition))

    def __str__(self):
        return "Transition to {state}{cond}".format(state=self.next_state,
            cond=self._condition_printing_suffix())

    exec_method = six.moves.intern("exec_StateTransition")


class FailStep(Instruction):
    """
    """

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return get_variables(self.condition)

    def map_expressions(self, mapper):
        return self.copy(condition=mapper(self.condition))

    def __str__(self):
        return "FailStep{cond}".format(cond=self._condition_printing_suffix())

    exec_method = six.moves.intern("exec_FailStep")


# }}}


# {{{ code container

class TimeIntegratorState(RecordWithoutPickling):
    """
    .. attribute:: depends_on

        a list of instruction IDs that need to be accomplished
        for successful execution of one round of this state

    .. attribute:: next_state

        name of the next state after this one, if no other state
        is specified by the user.
    """

    @classmethod
    def from_cb(cls, cb, next_state):
        """
        :arg cb: A :class:`CodeBuilder` instance
        :arg next_state: The name of the default next state
        """
        return cls(depends_on=cb.state_dependencies, next_state=next_state)

    def __init__(self, depends_on, next_state):
        super(TimeIntegratorState, self).__init__(
                depends_on=depends_on,
                next_state=next_state)


class TimeIntegratorCode(RecordWithoutPickling):
    """
    .. attribute:: instructions

        is a list of Instruction instances, in no particular
        order

    .. attribute:: states

        is a map from time integrator state names to :class:`TimeIntegratorState`
        instances

    .. attribute:: initial_state

        the name of the starting state
    """

    @classmethod
    def create_with_steady_state(cls, dep_on, instructions):
        states = {'main': TimeIntegratorState(dep_on, next_state='main')}
        return cls(instructions, states, 'main')

    @classmethod
    def create_with_init_and_step(cls, initialization_dep_on,
                                  step_dep_on, instructions):
        states = {}
        states['initialization'] = TimeIntegratorState(
                initialization_dep_on,
                next_state='primary')

        states['primary'] = TimeIntegratorState(
                step_dep_on,
                next_state='primary')

        return cls(instructions, states, 'initialization')

    def __init__(self, instructions, states, initial_state):
        assert not isinstance(states, list)
        RecordWithoutPickling.__init__(self, instructions=instructions,
                                       states=states,
                                       initial_state=initial_state)

    @property
    @memoize_method
    def id_to_insn(self):
        return dict((insn.id, insn)
                for insn in self.instructions)

    # {{{ identifier wrangling

    def get_insn_id_generator(self):
        from pytools import UniqueNameGenerator
        return UniqueNameGenerator(
                set(insn.id for insn in self.instructions))

    def existing_var_names(self):
        result = set()
        for insn in self.instructions:
            result.update(insn.get_assignees())
            result.update(insn.get_read_variables())

        return result

    def get_var_name_generator(self):
        from pytools import UniqueNameGenerator
        return UniqueNameGenerator(self.existing_var_names())

    # }}}

    def __str__(self):
        lines = []

        def print_insn(insn):
            if insn.id in printed_insn_ids:
                return
            printed_insn_ids.add(insn.id)

            for dep_id in insn.depends_on:
                print_insn(self.id_to_insn[dep_id])

            lines.append("    {%s} %s" % (insn.id, insn))

        for state_name, state in six.iteritems(self.states):
            printed_insn_ids = set()

            lines.append("STATE %s" % state_name)

            for root_id in state.depends_on:
                print_insn(self.id_to_insn[root_id])

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

            if not target.evaluate_condition(insn):
                continue

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

            from leap.vm.expression import parse

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

    def __call__(self, assignee, expression, loops=[]):
        """Generate code for an assignment.

        *assignee* may be a variable or a subscript (if referring to an
        array)
        """

        from leap.vm.expression import parse

        if isinstance(assignee, str):
            assignee = parse(assignee)

        if isinstance(expression, str):
            expression = parse(expression)

        new_loops = []
        for ident, start, stop in loops:
            if isinstance(start, str):
                start = parse(start)
            if isinstance(stop, str):
                stop = parse(stop)
            new_loops.append((ident, start, stop))

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
            raise ValueError("asignee (left-hand side) must be either a variable "
                    "or a subscribted variable, not '%s'"
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
        for assignee in inst.get_assignees():
            # Warn about potential ordering of assignments that may
            # be unexpected by the user.
            if assignee in context.used_variables:
                raise ValueError("write after use of " + assignee +
                                 " in the same block")
            if assignee in context.definition_map:
                raise ValueError("multiple assignments to " + assignee)

        # Create the set of dependencies based on the set of used
        # variables.
        for used_variable in inst.get_read_variables():
            if used_variable in context.definition_map:
                dependencies.add(context.definition_map[used_variable])

        # Add the condition to the instruction.
        # Update context and global information.
        context.context_instruction_ids.add(inst_id)
        context.definition_map.update((assignee, inst_id)
                                      for assignee in inst.get_assignees())
        context.used_variables |= inst.get_read_variables()
        self._all_var_names |= inst.get_assignees()
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

        from leap.vm.expression import parse

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

    def raise_(self, error):
        self.fence()
        self._add_inst_to_context(Raise(error))

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

        if 0:
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

    for (insn_1, insn_2), annot in six.iteritems(annotation_dep_graph):
            lines.append(
                    "%s -> %s  [label=\"%s\", style=dashed]"
                    % (insn_2, insn_1, annot))

    for i, (name, state) in enumerate(six.iteritems(code.states)):
        lines.append("subgraph cluster_%d { label=\"%s\"" % (i, name))
        for dep in state.depends_on:
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
