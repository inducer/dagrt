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
from leap.vm.language import Instruction, YieldState, StateTransition


# {{{ verifier

class InstructionDAGVerifier(object):
    """Verifies that code is well-formed.

    .. attribute:: warnings

        is a human-readable list of warnings detected by the verifier.

    .. attribute:: errors

        is a human-readable list of errors detected by the verifier.
    """

    def __init__(self, instructions, state_names, dependency_lists):
        """
        :arg instructions: A set of instructions to verify
        :arg state_names: The list of state names
        :arg dependency_lists: A list of sets of instruction ids. Each set of
            instruction ids represents the dependencies for a state.
        """
        warnings = []
        errors = []

        # TODO: Warn about conditions that may have side effects.
        # Eg:
        # WARNING: The condition expression for this function includes a call to
        # an external function.  leap assumes that evaluating conditional
        # expressions does not produce side effects.  If calling the function
        # does produce a side effect, the behavior of the code may not be what
        # you expect.

        # TODO: Make error messages more useful. Eg. name the offending
        # instruction.

        if not self._verify_instructions_well_typed(instructions):
            errors += ['Instructions are not well formed.']
        elif not self._verify_state_transitions(instructions, state_names):
            errors += ['A state referenced by a transition instruction'
                       ' does not exit.']
        elif not self._verify_all_dependencies_exist(instructions,
                                                     *dependency_lists):
            errors += ['Code is missing a dependency.']
        elif not self._verify_no_circular_dependencies(instructions):
            errors += ['Code has circular dependencies.']

        self.warnings = warnings
        self.errors = errors

    def _verify_instructions_well_typed(self, instructions):
        """Ensure that all instructions are of the expected format."""
        for inst in instructions:
            # TODO: Be strict about the correctness of the input.
            if not isinstance(inst, Instruction):
                return False
        return True

    def _verify_state_transitions(self, instructions, states):
        """Ensure that states referenced by StateTransition exist."""
        for inst in instructions:
            if isinstance(inst, StateTransition):
                if inst.next_state not in states:
                    return False
        return True

    def _verify_all_dependencies_exist(self, instructions, *dependency_lists):
        """Ensure that all instruction dependencies exist."""
        ids = set(inst.id for inst in instructions)
        for inst in instructions:
            deps = set(inst.depends_on)
            if not deps <= ids:
                return False
        for dependency_list in dependency_lists:
            if not set(dependency_list) <= ids:
                return False
        return True

    def _verify_no_circular_dependencies(self, instructions):
        """Ensure that there are no circular dependencies among the instructions."""
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
                        return False
                    stack.append(id_to_instruction[neighbor])
            else:
                if top.id in visiting:
                    visiting.remove(top.id)
                stack.pop()
        return True


class CodeGenerationError(Exception):
    def __init__(self, errors):
        self.errors = errors

    def __str__(self):
        return 'Errors encountered in input to code generator.\n' + \
            '\n'.join(self.errors)


class CodeGenerationWarning(UserWarning):
    pass


def verify_code(code):
    """Verify that the DAG is well-formed."""
    from .analysis import InstructionDAGVerifier
    verifier = InstructionDAGVerifier(
            code.instructions,
            [state for state in code.states.keys()],
            [state.depends_on
                for state in code.states.values()])

    if verifier.errors:
        raise CodeGenerationError(verifier.errors)
    if verifier.warnings:
        # Python comes with a facility to silence/filter warnigns,
        # no need to replicate that functionality.

        from warnings import warn
        for warning in verifier.warnings:
            warn(warning, CodeGenerationWarning)

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


def collect_function_names_from_dag(dag):
    fnc = _FunctionNameCollector()

    result = set()

    def mapper(expr):
        result.update(fnc(expr))
        return expr
    for insn in dag.instructions:
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
