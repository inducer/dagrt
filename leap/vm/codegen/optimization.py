"""Optimization passes"""

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

from .ir import AssignInst, BranchInst, JumpInst, ReturnInst
from .analysis import ReachingDefinitions
from pytools import one


class Optimizer(object):
    """Performs optimizations on the code in a function."""

    def __call__(self, function):
        cfg_simplify = ControlFlowGraphSimplifier()
        adce = AggressiveDeadCodeElimination()

        iterations = 0
        changed = True
        while changed and iterations < 5:
            # Attempt to iterate until convergence.
            changed = False
            changed |= cfg_simplify(function)
            changed |= adce(function)
            iterations += 1

        return function


class ControlFlowGraphSimplifier(object):
    """Performs simplification optimizations on the control-flow graph."""

    def __call__(self, function):
        changed = False
        changed |= self.coalesce_jumps(function)
        changed |= self.discard_unreachable_blocks(function)
        changed |= self.merge_basic_blocks(function)
        return changed

    def discard_unreachable_blocks(self, control_flow_graph):
        """Search the control flow graph for reachable blocks by following
        actual edges. Remove all references to blocks that are unreachable."""
        reachable = set()
        stack = [control_flow_graph.start_block]
        while stack:
            top = stack.pop()
            if top not in reachable:
                reachable.add(top)
                stack.extend(top.code[-1].get_jump_targets())
        for block in reachable:
            block.successors &= reachable
            block.predecessors &= reachable
        all_blocks = {block for block in control_flow_graph}
        changed = reachable != all_blocks
        if changed:
            control_flow_graph.update()
        return changed

    def coalesce_jumps(self, control_flow_graph):
        """Bypass basic blocks that consist of a single jump instruction."""

        # Find and compute the targets of all blocks that are trivial jumps.
        trivial_jumps = {}
        for block in control_flow_graph.postorder():
            if self.block_is_trivial_jump(block):
                dest = one(block.successors)
                if dest in trivial_jumps:
                    trivial_jumps[block] = trivial_jumps[dest]
                else:
                    trivial_jumps[block] = dest

        changed = False

        # Update all blocks to bypass trivial jumps.
        for block in control_flow_graph:
            terminator = block.code[-1]
            if isinstance(terminator, JumpInst) and \
                    terminator.dest in trivial_jumps:
                new_dest = trivial_jumps[terminator.dest]
                block.delete_instruction(terminator)
                block.add_instruction(JumpInst(dest=new_dest))
                changed = True
            elif isinstance(terminator, BranchInst) and \
                    (terminator.on_true in trivial_jumps or
                    terminator.on_false in trivial_jumps):
                new_branch = BranchInst(condition=terminator.condition,
                                        on_true=terminator.on_true,
                                        on_false=terminator.on_false)
                if terminator.on_true in trivial_jumps:
                    new_branch.on_true = trivial_jumps[terminator.on_true]
                if terminator.on_false in trivial_jumps:
                    new_branch.on_false = trivial_jumps[terminator.on_false]
                block.delete_instruction(terminator)
                block.add_instruction(new_branch)
                changed = True

        if changed:
            control_flow_graph.update()
        return changed

    def block_is_trivial_jump(self, block):
        return len(block.code) == 1 and len(block.successors) == 1

    def merge_basic_blocks(self, control_flow_graph):
        """Merge basic blocks that can be trivially combined."""

        regions = []
        has_region = set()
        changed = False

        for block in control_flow_graph.reverse_postorder():

            if block in has_region:
                continue
            region = [block]

            # Extract a maximal basic block.
            while len(block.successors) == 1:
                block = one(block.successors)
                if len(block.predecessors) == 1:
                    region.append(block)
                else:
                    break

            has_region.update(region)
            regions.append(region)

        # Merge blocks and update references.
        for region in regions:

            if len(region) == 1:
                continue

            # Create the new merged block.
            code = []
            for block in region:
                code += block.code[:-1]
                if block == region[-1]:
                    code.append(block.code[-1])
                block.clear()
            header = region[0]
            header.add_instructions(code)
            changed = True

        if changed:
            control_flow_graph.update()
        return changed


class AggressiveDeadCodeElimination(object):
    """Removes dead code."""

    def __call__(self, control_flow_graph):
        reaching_definitions = ReachingDefinitions(control_flow_graph)

        # Find all trivially essential instructions.
        essential = set()
        for block in control_flow_graph:
            essential |= \
                {inst for inst in block if self.is_trivially_essential(inst)}

        # Working backwards from the set of trivially essential instructions,
        # discover all essential instructions.
        worklist = list(essential)
        while worklist:
            inst = worklist.pop()
            dependencies = self.get_dependent_instructions(
                inst, reaching_definitions)
            for dependency in dependencies:
                if dependency not in essential:
                    essential.add(dependency)
                    worklist.append(dependency)

        # Remove instructions that are not marked as essential.
        changed = False
        for block in control_flow_graph:
            to_delete = [instr for instr in block if instr not in essential]
            block.delete_instructions(to_delete)
            changed |= len(to_delete) > 0

        return changed

    def get_dependent_instructions(self, inst, reaching_definitions):
        definitions = reaching_definitions.get_reaching_definitions(inst)
        variables = inst.get_used_variables()
        insts = {pair[1] for pair in definitions if pair[0] in variables}
        return insts

    def is_trivially_essential(self, inst):
        if isinstance(inst, ReturnInst):
            # All return instructions are essential.
            return True
        elif isinstance(inst, BranchInst) or isinstance(inst, JumpInst):
            # All control flow instructions are essential. This is a
            # pessimistic assumption and may be improved upon if the
            # worklist algorithm used control dependence to discover
            # the set of essential control flow instructions.
            return True
        elif isinstance(inst, AssignInst):
            symbol_table = inst.block.symbol_table
            # All assignments to state variables are essential.
            for assignee in inst.get_defined_variables():
                if symbol_table[assignee].is_global:
                    return True
        else:
            return False
