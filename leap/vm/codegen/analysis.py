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

from pytools import memoize_method
from leap.vm.language import Instruction, If
from .graphs import InstructionDAGIntGraph


class InstructionDAGVerifier(object):
    """Verifies that code is well-formed."""

    def __init__(self, instructions, *dependency_lists):
        warnings = []
        errors = []

        if not self.verify_instructions_well_typed(instructions):
            errors += ['Instructions are not well formed.']
        elif not self.verify_all_dependencies_exist(instructions,
                                                    *dependency_lists):
            errors += ['Code is missing a dependency.']
        elif not self.verify_no_circular_dependencies(instructions):
            errors += ['Code has circular dependencies.']

        self.warnings = warnings
        self.errors = errors

    def verify_instructions_well_typed(self, instructions):
        """Ensure that all instructions are of the expected format."""
        for inst in instructions:
            # TODO: To what extent should the verifier check the correctness
            # of the input?
            if not isinstance(inst, Instruction):
                return False
        return True

    def verify_all_dependencies_exist(self, instructions, *dependency_lists):
        """Ensure that all instruction dependencies exist."""
        ids = set(inst.id for inst in instructions)
        for inst in instructions:
            deps = set(inst.depends_on)
            if isinstance(inst, If):
                deps |= set(inst.then_depends_on)
                deps |= set(inst.else_depends_on)
            if not deps <= ids:
                return False
        for dependency_list in dependency_lists:
            if not set(dependency_list) <= ids:
                return False
        return True

    def verify_no_circular_dependencies(self, instructions):
        """Ensure that there are no circular dependencies among the
        instructions.
        """
        graph = InstructionDAGIntGraph(instructions)
        visited = set()
        visiting = set()
        stack = list(graph)
        while stack:
            top = stack[-1]
            if top not in visited:
                visited.add(top)
                visiting.add(top)
                for dep in graph[top]:
                    if dep in visiting:
                        return False
                    else:
                        stack.append(dep)
            else:
                visiting.discard(top)
                stack.pop()
        return True


class ReachingDefinitions(object):
    """Performs a reaching definitions analysis and computes use-def chains."""

    def __init__(self, control_flow_graph):
        # A definition is a pair (variable, instruction) representing a
        # variable name and the instruction which defines the variable.

        def_in = {}
        def_out = {}
        def_gen = {}
        def_kill = {}

        # Initialize the gen, kill, and definition sets for dataflow analysis.
        for block in control_flow_graph:
            gen, kill = self.get_gen_and_kill_sets(block, len(block))
            def_gen[block] = gen
            def_kill[block] = kill
            def_in[block] = set()
            def_out[block] = set()

        # Perform the reaching definitions analysis.
        changed = True
        while changed:
            # Iterate until convergence.
            changed = False

            for block in control_flow_graph.reverse_postorder():
                # Compute the set of definitions that reach the entry of the
                # block and the exit.
                reach = set()
                for predecessor in block.predecessors:
                    reach |= def_out[predecessor]
                changed |= len(reach) > len(def_in[block])
                def_in[block] = reach

                kill = def_kill[block]
                reach_out = self.remove_killed(reach, kill)

                reach_out |= def_gen[block]
                changed |= len(reach_out) > len(def_out[block])

                def_out[block] = reach_out

            # If the graph is acyclic then only a single iteration is required.
            if control_flow_graph.is_acyclic():
                break

        self.def_in = def_in
        self.def_out = def_out

    def get_gen_and_kill_sets(self, block, point):
        """Return the gen and kill sets."""
        last_def = {}
        for pos, inst in enumerate(block.code):
            if pos == point:
                break
            for name in inst.get_defined_variables():
                last_def[name] = inst
        return (set(last_def.iteritems()), set(last_def.iterkeys()))

    def remove_killed(self, definitions, kill):
        """Return the result of removing all definitions that are killed."""
        return set(pair for pair in definitions if pair[0] not in kill)

    @memoize_method
    def get_reaching_definitions(self, instruction):
        """Returns the set of all definitions that reach the instruction on some
        execution path."""
        block = instruction.block
        index = block.code.index(instruction)
        gen, kill = self.get_gen_and_kill_sets(block, index)
        return gen | self.remove_killed(self.def_in[block], kill)
