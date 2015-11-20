"""Analysis passes"""

__copyright__ = "Copyright (C) 2014 Matt Wala"

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

from pymbolic.mapper import Collector
from dagrt.language import YieldState, StateTransition, AssignFunctionCall


# {{{ verifier


def _quote(string):
    return "\"{}\"".format(string)


def verify_state_transitions(instructions, states, errors):
    """
    Ensure that states referenced by StateTransition exist.

    :arg instructions: A set of instructions to verify
    :arg states: A map from state names to states
    :arg errors: An error list to which new errors get appended
    """
    state_names = [key for key in states.keys()]
    for inst in instructions:
        if not isinstance(inst, StateTransition):
            continue
        if inst.next_state not in state_names:
            errors.append(
                "State \"{}\" referenced by instruction \"{}\" not found"
                .format(inst.next_state, inst))


def verify_all_dependencies_exist(instructions, states, errors):
    """
    Ensure that all instruction dependencies exist.

    :arg instructions: A set of instructions to verify
    :arg states: A map from state names to states
    :arg errors: An error list to which new errors get appended
    """
    ids = set(inst.id for inst in instructions)

    # Check instructions
    for inst in instructions:
        deps = set(inst.depends_on)
        if not deps <= ids:
            errors.extend(
                ["Dependency \"{}\" referenced by instruction \"{}\" not found"
                 .format(dep_name, inst) for dep_name in deps - ids])

    # Check states.
    import six
    for state_name, state in six.iteritems(states):
        deps = set(state.depends_on)
        if not deps <= ids:
            errors.extend(
                ["Dependencies {} referenced by state \"{}\" not found"
                 .format(", ".join(_quote(dep) for dep in ids - deps),
                         state_name)])


def verify_no_circular_dependencies(instructions, errors):
    """
    Ensure that there are no circular dependencies among the instructions.

    :arg instructions: A set of instructions to verify
    :arg errors: An error list to which new errors get appended
    """
    id_to_instruction = dict((inst.id, inst) for inst in instructions)
    stack = list(instructions)
    visiting = set()
    visited = set()
    while stack:
        top = stack[-1]
        if top.id not in visited:
            visited.add(top.id)
            visiting.add(top.id)
            for neighbor in top.depends_on:
                if neighbor in visiting:
                    errors.append("Circular dependency chain found")
                    return
                stack.append(id_to_instruction[neighbor])
        else:
            if top.id in visiting:
                visiting.remove(top.id)
            stack.pop()


def verify_single_definition_cond_rule(instructions, errors):
    """
    Verify that <cond> variables are never redefined.

    :arg instructions: A set of instructions to verify
    :arg errors: An error list to which new errors get appended
    """
    cond_variables = {}

    for instruction in instructions:
        for varname in instruction.get_assignees():
            if not varname.startswith("<cond>"):
                continue
            if varname not in cond_variables:
                cond_variables[varname] = [instruction]
            else:
                cond_variables[varname].append(instruction)

    import six
    for varname, insts in six.iteritems(cond_variables):
        if len(insts) > 1:
            errors.append(
                "Conditional variable \"{}\" defined by multiple instructions: {}"
                .format(varname, ", ".join(_quote(str(inst)) for inst in insts)))


class CodeGenerationError(Exception):
    def __init__(self, errors):
        self.errors = errors

    def __str__(self):
        return 'Errors encountered in input to code generator.\n' + \
            '\n'.join(self.errors)


def verify_code(code):
    """Verify that the DAG is well-formed."""
    errors = []

    try:
        # Wrap in a try block, since some verifier passes may fail due to badly
        # malformed code.
        verify_all_dependencies_exist(code.instructions, code.states, errors)
        verify_no_circular_dependencies(code.instructions, errors)
        verify_state_transitions(code.instructions, code.states, errors)
        verify_single_definition_cond_rule(code.instructions, errors)
    except Exception as e:
        # Ensure there is at least one error to report.
        if len(errors) == 0:
            raise e

    if errors:
        raise CodeGenerationError(errors)

# }}}


# {{{ collect function names from DAG

class _FunctionNameCollector(Collector):

    def map_variable(self, expr):
        if expr.name.startswith("<func>"):
            return set([expr.name])
        return set()

    def map_call(self, expr):
        return (set([expr.function])
                | super(_FunctionNameCollector, self).map_call(expr))

    def map_call_with_kwargs(self, expr):
        return (set([expr.function])
                | super(_FunctionNameCollector, self).map_call_with_kwargs(expr))


def collect_function_names_from_dag(dag, no_expressions=False):
    """
    :arg no_expressions: Do not consider expressions when finding function names.
        This saves a bit of time if, for example,
        :func:`dagrt.codegen.transform.isolate_function_calls` has been called
        on *dag*.
    """
    fnc = _FunctionNameCollector()

    result = set()

    def mapper(expr):
        result.update(fnc(expr))
        return expr
    for insn in dag.instructions:
        if isinstance(insn, AssignFunctionCall):
            result.add(insn.function_id)

        if not no_expressions:
            insn.map_expressions(mapper)

    return result

# }}}


# {{{ collect time IDs from DAG

def collect_time_ids_from_dag(dag):
    result = set()

    for insn in dag.instructions:
        if isinstance(insn, YieldState):
            result.add(insn.time_id)

    return result

# }}}


# {{{ collect ODE component names from DAG

def collect_ode_component_names_from_dag(dag):
    result = set()

    for insn in dag.instructions:
        if isinstance(insn, YieldState):
            result.add(insn.component_id)

    return result

# }}}

# vim: foldmethod=marker
