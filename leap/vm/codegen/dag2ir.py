"""Turn the DAG representation of a timestepper into a flow graph"""

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

from leap.vm.language import Instruction, AssignExpression, AssignNorm, \
    AssignDotProduct, AssignSolvedRHS, AssignRHS, If, ReturnState, Raise, \
    FailStep
from .graphs import InstructionDAGIntGraph
from leap.vm.utils import peek, get_unique_name, is_state_variable
from .ir import SymbolTable, BasicBlock, ControlFlowGraph
from pymbolic import var


class Entry(Instruction):
    """Dummy entry point for the instruction DAG."""

    def __init__(self, **kwargs):
        Instruction.__init__(self, **kwargs)

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return frozenset()

    def __str__(self):
        return "Entry"

    exec_method = "exec_Entry"


class Exit(Instruction):
    """Dummy exit point for the instruction DAG."""

    def __init__(self, **kwargs):
        Instruction.__init__(self, **kwargs)

    def get_assignees(self):
        return frozenset()

    def get_read_variables(self):
        return frozenset()

    def __str__(self):
        return "Exit"

    exec_method = "exec_Exit"


class InstructionDAGEntryExitAugmenter(object):
    """Augments an instruction DAG with entry and exit instructions."""

    def __call__(self, instructions, instructions_dep_on):
        """Return a new, augmented set of instructions that include an entry
        and exit instruction. Every instruction depends on the entry
        instruction while the exit instruction depends on the set
        instructions_dep_on.
        """
        ids = set(inst.id for inst in instructions)
        entry_id = get_unique_name('entry', ids)
        exit_id = get_unique_name('exit', ids)
        import copy
        aug_instructions = copy.deepcopy(instructions)
        ent = Entry(id=entry_id)
        ex = Exit(id=exit_id, depends_on=[entry_id] + instructions_dep_on)
        aug_instructions.add(ex)
        for inst in aug_instructions:
            inst.depends_on = frozenset([entry_id] + list(inst.depends_on))
        aug_instructions.add(ent)
        return (aug_instructions, ent, ex)


class InstructionDAGExtractor(object):
    """Returns only the portion of the DAG necessary for satisfying the
    specified dependencies."""

    def __call__(self, dag, dependencies):
        graph = InstructionDAGIntGraph(dag)
        stack = list(map(graph.get_number_for_id, dependencies))
        reachable = set()
        while len(stack) > 0:
            top = stack.pop()
            if top not in reachable:
                reachable.add(top)
                for vertex in graph[top]:
                    stack.append(vertex)
        return set(map(graph.get_vertex_for_number, reachable))


class InstructionDAGPartitioner(object):
    """Partition a list of instructions into maximal straight line
    sequences with dependency information."""

    def __call__(self, instructions):
        inst_graph = InstructionDAGIntGraph(instructions)

        # The unconditional dependency relation is transitive. Prune as many
        # unconditional edges as we can while maintaining all dependencies.
        tr_graph = self.unconditional_transitive_reduction(inst_graph)

        # Construct maximal length straight line sequences of instructions that
        # respect all dependencies.
        num_block_graph, num_to_block = self.maximal_blocks(tr_graph,
                                                            inst_graph)

        # Convert the constructed sequences into lists of instruction ids.
        id_for_num = inst_graph.get_id_for_number
        to_id = lambda block: tuple(map(id_for_num, block))
        block_graph = dict([to_id(bl), map(to_id, bls)] for (bl, bls) in
                           num_block_graph.iteritems())
        inst_id_to_block = dict([id_for_num(i), to_id(bl)] for (i, bl) in
                                num_to_block.iteritems())

        return (block_graph, inst_id_to_block)

    def topological_sort(self, dag):
        """Returns a topological sort of the input DAG."""
        unvisited = set(dag)
        visiting = set()
        stack = []
        sort = []
        while len(unvisited) > 0:
            stack.append(peek(unvisited))
            while len(stack) > 0:
                top = stack[-1]
                if top in unvisited:
                    unvisited.remove(top)
                    visiting.add(top)
                    for dep in dag[top]:
                        stack.append(dep)
                else:
                    if top in visiting:
                        visiting.remove(top)
                        sort.append(top)
                    stack.pop()
        return sort

    def unconditional_transitive_reduction(self, dag):
        """Returns a transitive reduction of the unconditional portion of the
        input instruction DAG. Conditional edges are kept."""
        # Compute u -> v longest unconditional paths in the DAG.
        longest_path = dict(((u, v), 0 if u == v else -1) for u in dag
                            for v in dag)
        topo_sort = self.topological_sort(dag)
        topo_sort.reverse()
        for i, vertex in enumerate(topo_sort):
            for intermediate_vertex in topo_sort[i:]:
                if longest_path[(vertex, intermediate_vertex)] >= 0:
                    edges = dag.get_unconditional_edges(intermediate_vertex)
                    for successor in edges:
                        old = longest_path[(vertex, successor)]
                        new = 1 + longest_path[(vertex, intermediate_vertex)]
                        longest_path[(vertex, successor)] = max(old, new)

        # Keep only those unconditional u -> v edges such that
        # longestPath(u, v) = 1.
        reduction = {}
        for vertex in dag:
            reduction[vertex] = set()
            for successor in dag.get_unconditional_edges(vertex):
                if longest_path[(vertex, successor)] == 1:
                    reduction[vertex].add(successor)

        # Keep all conditional edges.
        for vertex in dag:
            for successor in dag.get_conditional_edges(vertex):
                reduction[vertex].add(successor)

        return reduction

    def maximal_blocks(self, dag, original_dag):
        """Returns a partition of the DAG into maximal blocks of straight-line
        pieces."""
        # Compute the inverse of the DAG.
        dag_inv = dict((u, set()) for u in dag)
        for vertex, successors in dag.iteritems():
            for successor in successors:
                dag_inv[successor].add(vertex)

        # Traverse the DAG extracting maximal straight line sequences into
        # blocks.
        topo_sort = self.topological_sort(dag)
        visited = set()
        blocks = set()
        inst_to_block = {}
        while len(topo_sort) > 0:
            instr = topo_sort.pop()
            if instr in visited:
                continue
            visited.add(instr)
            block = [instr]
            # Traverse down from instr.
            while len(dag[instr]) == 1:
                instr = peek(dag[instr])
                if len(dag_inv[instr]) == 1:
                    visited.add(instr)
                    block.append(instr)
                else:
                    break
            block.reverse()
            block = tuple(block)
            for i in block:
                inst_to_block[i] = block
            blocks.add(block)

        # Record the graph structure of the blocks.
        block_graph = {}
        # Get the unconditional dependencies of each instruction.
        for block in blocks:
            block_graph[block] = \
                set(inst_to_block[i] for i in dag[block[0]]
                    if i in original_dag.get_unconditional_edges(block[0]))
        return (block_graph, inst_to_block)


class FlagAnalysis(object):
    """Keeps track of the values of a set of boolean flags."""

    def __init__(self, flags):
        """Creates a flag analysis object that keeps track of the given set of
        flags."""
        self.all_flags = set(flags)
        self.must_be_true = set()
        self.must_be_false = set()

    def set_true(self, flag):
        """Returns a new flag analysis object with the given flag set to true.
        """
        assert flag in self.all_flags
        import copy
        new_fa = copy.deepcopy(self)
        new_fa.must_be_true.add(flag)
        new_fa.must_be_false.discard(flag)
        return new_fa

    def set_false(self, flag):
        """Returns a new flag analysis object with the given flag set to false.
        """
        assert flag in self.all_flags
        import copy
        new_fa = copy.deepcopy(self)
        new_fa.must_be_false.add(flag)
        new_fa.must_be_true.discard(flag)
        return new_fa

    def is_definitely_true(self, flag):
        """Determines if the flag must be set to true."""
        assert flag in self.all_flags
        return flag in self.must_be_true

    def is_definitely_false(self, flag):
        """Determines if the flag must be set to false."""
        assert flag in self.all_flags
        return flag in self.must_be_false

    def __and__(self, other):
        """Returns a new flag analysis that represents the conjunction of the
        inputs."""
        assert isinstance(other, FlagAnalysis)
        assert self.all_flags == other.all_flags
        new_fa = FlagAnalysis(self.all_flags)
        new_fa.must_be_true = self.must_be_true & other.must_be_true
        new_fa.must_be_false = self.must_be_false & other.must_be_false
        return new_fa


class ControlFlowGraphAssembler(object):
    """Constructs a control flow graph from an instruction DAG."""

    def __call__(self, instructions, instructions_dep_on):
        # Add Entry and Exit instructions to the DAG.
        augmenter = InstructionDAGEntryExitAugmenter()
        aug_instructions, ent, ex = \
            augmenter(instructions, instructions_dep_on)

        # Partition the DAG into maximal straight line instruction blocks.
        partitioner = InstructionDAGPartitioner()
        block_graph, inst_id_to_block = partitioner(aug_instructions)

        # Save the block graph.
        self.block_graph = block_graph
        self.inst_id_to_inst = dict([i.id, i] for i in aug_instructions)
        self.inst_id_to_block = inst_id_to_block

        # Set up the symbol and flag tables.
        self.initialize_symbol_table(aug_instructions, block_graph)
        self.initialize_flags(block_graph)

        # Initialize a new variable to hold the return value.
        self.return_val = self.symbol_table.get_fresh_variable_name('retval')
        self.symbol_table.add_variable(self.return_val, 'is_return_val')

        # Find the exit block and create a new basic block out of it.
        exit_block = inst_id_to_block[ex.id]

        # Create the initial basic block.
        self.basic_block_count = 0
        entry_bb = self.get_entry_block()

        # Set up the initial flag analysis.
        flag_names = set(self.flags.itervalues())
        flag_analysis = FlagAnalysis(flag_names)
        flag_analysis.must_be_false = set(flag_names)

        # Create the control flow graph.
        end_bb, flag_analysis = self.process_block(exit_block, entry_bb,
                                                   flag_analysis)

        if not end_bb.terminated:
            end_bb.add_unreachable()

        return ControlFlowGraph(entry_bb)

    def new_basic_block(self):
        """Create a new, empty basic block with a unique number."""
        number = self.basic_block_count
        self.basic_block_count += 1
        return BasicBlock(number, self.symbol_table)

    def initialize_flags(self, block_graph):
        """Create the flags for the blocks."""
        self.flags = {}
        block_count = 0
        symbol_table = self.symbol_table
        # Create a flag for each block and insert into the symbol table.
        for block in block_graph:
            block_id = block_count
            block_count += 1
            flag = symbol_table.get_fresh_variable_name('flag_%d' % block_id)
            self.flags[block] = flag
            symbol_table.add_variable(flag, 'is_flag')

    def initialize_symbol_table(self, aug_instructions, block_graph):
        """Create a new symbol table and add all variables that have been
        used in the instruction list to the symbol table."""

        symbol_table = SymbolTable()

        # Get a list of all used variable names and right hand sides.
        var_names = set()
        rhs_names = set()
        for inst in aug_instructions:
            var_names |= set(inst.get_assignees())
            var_names |= set(inst.get_read_variables())
            if isinstance(inst, AssignRHS):
                rhs_names.add(inst.component_id)

        # Create a symbol table entry for each variable.
        for var_name in var_names:
            symbol_table.add_variable(var_name)
            if is_state_variable(var_name):
                symbol_table[var_name].is_global = True

        # Record the RHSs.
        symbol_table.rhs_names = rhs_names

        self.symbol_table = symbol_table

    def get_entry_block(self):
        """Create the entry block of the control flow graph."""
        start_bb = self.new_basic_block()
        # Initialize the flag variables.
        for flag in self.flags.itervalues():
            start_bb.add_assignment((flag, False))
        # Initialize the return value.
        start_bb.add_assignment((self.return_val, 0))
        return start_bb

    def process_block_sequence(self, block_sequence, top_bb, flag_analysis):
        """Produce a control flow subgraph that corresponds to a sequence of
        instruction blocks.
        """

        if len(block_sequence) == 0:
            return (top_bb, flag_analysis)

        main_bb = top_bb
        for block in block_sequence:
            main_bb, flag_analysis = self.process_block(block, main_bb,
                                                        flag_analysis)

        return (main_bb, flag_analysis)

    def process_block(self, inst_block, top_bb, flag_analysis):
        """Produce the control flow subgraph corresponding to a block of
        instructions."""

        get_block_set = lambda inst_set: \
            map(self.inst_id_to_block.__getitem__, inst_set)

        # Check the flag analysis to see if we need to compute the block.
        flag = self.flags[inst_block]

        if flag_analysis.is_definitely_true(flag):
            return (top_bb, flag_analysis)

        needs_flag = not flag_analysis.is_definitely_false(flag)

        # Process all dependencies.
        dependencies = self.block_graph[inst_block]
        main_bb, flag_analysis = \
            self.process_block_sequence(dependencies, top_bb, flag_analysis)

        if needs_flag:
            # Add code to check and set the flag for the block.
            new_main_bb = self.new_basic_block()
            merge_bb = self.new_basic_block()
            # Add a jump to the appropriate block from the top block
            from pymbolic.primitives import LogicalNot
            main_bb.add_branch(LogicalNot(var(flag)), new_main_bb, merge_bb)
            # Set the current block being built
            main_bb = new_main_bb

        for instruction_id in inst_block:
            instruction = self.inst_id_to_inst[instruction_id]

            if isinstance(instruction, Entry):
                continue

            elif isinstance(instruction, Exit):
                main_bb.add_return(var(self.return_val))
                break

            elif isinstance(instruction, If):
                # Get the destination instruction blocks.
                then_blocks = get_block_set(instruction.then_depends_on)
                else_blocks = get_block_set(instruction.else_depends_on)

                # Create basic blocks for then, else, and merge point.
                then_bb = self.new_basic_block()
                else_bb = self.new_basic_block()
                then_else_merge_bb = self.new_basic_block()

                # Emit basic blocks for then and else components.
                end_then_bb, then_flag_analysis = self.process_block_sequence(
                    then_blocks, then_bb, flag_analysis)
                end_else_bb, else_flag_analysis = self.process_block_sequence(
                    else_blocks, else_bb, flag_analysis)

                # Emit branch to then and else blocks.
                main_bb.add_branch(instruction.condition, then_bb, else_bb)

                # Emit branches to merge point.
                end_then_bb.add_jump(then_else_merge_bb)
                end_else_bb.add_jump(then_else_merge_bb)

                # Set the current basic block to be the merge point.
                flag_analysis = then_flag_analysis & else_flag_analysis
                main_bb = then_else_merge_bb

            elif isinstance(instruction, ReturnState):
                return_value = (instruction.time, instruction.time_id,
                                instruction.component_id,
                                instruction.expression)
                main_bb.add_assignment((self.return_val, return_value))

            elif isinstance(instruction, AssignExpression) or \
                    isinstance(instruction, AssignRHS) or \
                    isinstance(instruction, AssignNorm) or \
                    isinstance(instruction, AssignSolvedRHS) or \
                    isinstance(instruction, AssignDotProduct):
                main_bb.add_assignment(instruction)

            elif isinstance(instruction, Raise):
                pass

            elif isinstance(instruction, FailStep):
                pass

        if not main_bb.terminated:
            main_bb.add_assignment((flag, True))
            if needs_flag:
                main_bb.add_jump(merge_bb)
                main_bb = merge_bb

        flag_analysis = flag_analysis.set_true(flag)
        return (main_bb, flag_analysis)
