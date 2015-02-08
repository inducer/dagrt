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
from pytools import one

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
            id = six.moves.intern(id)
        depends_on = frozenset(kwargs.pop("depends_on", []))
        RecordWithoutPickling.__init__(self,
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

    def visit_expressions(self, visitor):
        pass

    def map_expressions(self, mapper):
        return self.copy()

    def __str__(self):
        return 'nop'

    exec_method = six.moves.intern("exec_Nop")


# {{{ assignments


class AssignSolved(Instruction):
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
                 solver_id, id=None, depends_on=frozenset()):
        Instruction.__init__(self, assignees=assignees,
                             solve_variables=solve_variables,
                             expressions=expressions,
                             other_params=other_params,
                             solver_id=solver_id,
                             id=id,
                             depends_on=depends_on)

    exec_method = six.moves.intern("exec_AssignSolved")

    def get_assignees(self):
        return frozenset(self.assignees)

    def get_read_variables(self):
        # Variables can be read by:
        #  1. expressions (except for those in solve_variables)
        #  2. values in other_params
        from itertools import chain
        flatten = lambda iter_arg: chain(*list(iter_arg))
        variables = frozenset()
        variables |= set(flatten(get_variables(expr) for expr in self.expressions))
        variables -= set(self.solve_variables)
        variables |= set(flatten(get_variables(expr) for expr
                                 in self.other_params.values()))
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
        return "\n".join(lines)


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
        return get_variables(self.expression)

    def map_expressions(self, mapper):
        return self.copy(
                expression=mapper(self.expression))

    def __str__(self):
        return "%s <- %s" % (self.assignee, self.expression)

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
        return get_variables(self.expression)

    def map_expressions(self, mapper):
        return self.copy(
                expression=mapper(self.expression))

    def __str__(self):
        return "Ret %s at %s as %s" % (
                self.expression,
                self.time_id,
                self.component_id)

    exec_method = six.moves.intern("exec_YieldState")


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

    def map_expressions(self, mapper):
        return self.copy()

    def __str__(self):
        result = "Raise %s" % self.error_condition.__name__

        if self.error_message:
            result += repr(self.error_message)

        return result

    exec_method = six.moves.intern("exec_Raise")


class StateTransition(Instruction):
    """
    .. attribute:: next_state

        The name of the next state to enter
    """

    def __init__(self, next_state, id=None, depends_on=frozenset()):
        Instruction.__init__(self, next_state=next_state, id=id,
                             depends_on=depends_on)

    def get_assignees(self):
        return frozenset()

    get_read_variables = get_assignees

    def map_expressions(self, mapper):
        return self.copy()

    def __str__(self):
        return "Transition to " + self.next_state

    exec_method = six.moves.intern("exec_StateTransition")


class FailStep(Instruction):
    """
    """

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return frozenset()

    def map_expressions(self, mapper):
        return self.copy()

    def __str__(self):
        return "FailStep"

    exec_method = six.moves.intern("exec_FailStep")


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
        return get_variables(self.condition)

    def map_expressions(self, mapper):
        return self.copy(
                condition=mapper(self.condition))

    def __str__(self):
        return "If %s" % self.condition

    exec_method = six.moves.intern("exec_If")

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
        :arg cb: A :class:`NewCodeBuilder` instance
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
        self.generated_id_set = set()
        self.var_set = set()
        self.generated_var_set = set()

        self._instructions = []
        self.build_group = []

    def fresh_var_name(self, prefix):
        """Return a variable name that is guaranteed not to be in use and not
        to be generated in the future."""
        from pytools import generate_unique_names
        for possible_var in generate_unique_names(str(prefix)):
            if possible_var not in self.var_set \
                    and possible_var not in self.generated_var_set:
                self.generated_var_set.add(possible_var)
                return possible_var

    def fresh_insn_id(self, prefix):
        """Return an instruction name that is guaranteed not to be in use and
        not to be generated in the future."""
        from pytools import generate_unique_names
        for possible_id in generate_unique_names(prefix):
            if possible_id not in self.id_set and possible_id not in \
                    self.generated_id_set:
                self.generated_id_set.add(possible_id)
                return possible_id

    def add_and_get_ids(self, *insns):
        new_ids = []
        for insn in insns:
            set_attrs = {}
            if not hasattr(insn, "id") or insn.id is None:
                set_attrs["id"] = self.fresh_insn_id("insn")
            else:
                if insn.id in self.id_set:
                    raise ValueError("duplicate ID")

            if not hasattr(insn, "depends_on"):
                set_attrs["depends_on"] = frozenset()

            if set_attrs:
                insn = insn.copy(**set_attrs)

            self.build_group.append(insn)
            new_ids.append(insn.id)

            self.var_set |= insn.get_assignees()
            # self.var_set |= insn.get_read_variables()

        # For exception safety, only make state change at end.
        self.id_set.update(new_ids)
        return new_ids

    def infer_single_writer_dependencies(self, exclude=[]):
        """Change the current :attr:`build_group` to one with reader
        dependencies added in situations where a only single instruction writes
        one variable.
        """

        for element in exclude:
            assert isinstance(element, str)

        var_to_writers = {}
        for insn in self.build_group:
            for wvar in insn.get_assignees():
                var_to_writers.setdefault(wvar, []).append(insn)

        # toss out variables written more than once
        for wvar in list(six.iterkeys(var_to_writers)):
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
        for insn in self.build_group:
            for dep in insn.depends_on:
                if dep not in self.id_set:
                    raise ValueError("unknown dependency id: %s" % dep)

        self._instructions.extend(self.build_group)
        del self.build_group[:]

    @property
    def instructions(self):
        if self.build_group:
            raise ValueError("attempted to get instructions while "
                    "build group is uncommitted")

        return self._instructions

# }}}


class NewCodeBuilder(object):

    class Context(RecordWithoutPickling):
        """
        Attributes:
        """
        def __init__(self, lead_instruction_ids=[], variable_map={},
                     used_variables=[]):
            RecordWithoutPickling.__init__(self,
                lead_instruction_ids=frozenset(lead_instruction_ids),
                context_instruction_ids=set(lead_instruction_ids),
                variable_map=dict(variable_map),
                used_variables=set(used_variables))

    def __init__(self, label="state"):
        self.label = label
        self._instruction_map = {}
        self._instruction_count = 0
        self._contexts = []
        self._all_var_names = set()
        self._all_generated_var_names = set()

    def fence(self):
        """
        Enter a new logical block of instructions. Force all prior
        instructions to execute before subsequent ones.
        """
        self._make_new_context_with_inst(Nop())

    @contextmanager
    def if_(self, *condition_arg):
        """Create a new block that is conditionally executed."""
        if len(condition_arg) == 1:
            condition = condition_arg[0]
        elif len(condition_arg) == 3:
            from pymbolic.primitives import Comparison
            condition = Comparison(*condition_arg)
        else:
            raise ValueError("Unrecognized condition expression")
        # Create a conditional instruction: then and else are empty.
        conditional = If(condition=condition, then_depends_on=[],
                         else_depends_on=[])
        conditional_id = self._make_new_context_with_inst(conditional)
        self._contexts.append(NewCodeBuilder.Context())
        yield
        # Update then_depends_on.
        then_context = self._contexts.pop()
        self._instruction_map[conditional_id] = \
            self._instruction_map[conditional_id].copy(
            then_depends_on=list(then_context.context_instruction_ids))

    @contextmanager
    def else_(self):
        """Create the "else" portion of a conditionally executed
        block.
        """
        conditional_id = one(self._contexts[-1].lead_instruction_ids)
        assert isinstance(self._instruction_map[conditional_id], If)
        self._contexts.append(NewCodeBuilder.Context())
        yield
        else_context = self._contexts.pop()
        self._instruction_map[conditional_id] = \
            self._instruction_map[conditional_id].copy(
            else_depends_on=list(else_context.context_instruction_ids))
        self.fence()

    def _next_instruction_id(self):
        self._instruction_count += 1
        return self.label + "_" + str(self._instruction_count)

    def __call__(self, assignee, expression):
        """Assign a value."""
        from pymbolic.primitives import Variable
        assert isinstance(assignee, Variable)
        self._add_inst_to_context(AssignExpression(
                assignee=assignee.name,
                expression=expression))

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
            if assignee in context.variable_map:
                raise ValueError("multiple assignments to " + assignee)
        # Create the set of dependencies based on the set of used
        # variables.
        for used_variable in inst.get_read_variables():
            if used_variable in context.variable_map:
                dependencies.add(context.variable_map[used_variable])
        # Update context and global information.
        context.context_instruction_ids.add(inst_id)
        context.variable_map.update((assignee, inst_id)
                                    for assignee in inst.get_assignees())
        context.used_variables |= inst.get_read_variables()
        self._all_var_names |= inst.get_assignees()
        self._instruction_map[inst_id] = \
            inst.copy(id=inst_id, depends_on=list(dependencies))
        return inst_id

    def _make_new_context_with_inst(self, inst):
        assert isinstance(inst, (Nop, If))
        inst_id = self._next_instruction_id()
        context = self._contexts[-1]
        new_context = NewCodeBuilder.Context(
            lead_instruction_ids=[inst_id],
            used_variables=inst.get_read_variables())
        self._instruction_map[inst_id] = \
            inst.copy(id=inst_id,
                      depends_on=list(context.context_instruction_ids))
        self._contexts[-1] = new_context
        return inst_id

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
        self._contexts.append(NewCodeBuilder.Context())
        return self

    def __exit__(self, *ignored):
        self.fence()
        self.state_dependencies = list(self._contexts[-1].lead_instruction_ids)
        self.instructions = set(self._instruction_map.values())


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
