from __future__ import division, with_statement

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

from leap.vm.language import Instruction
from leap.vm.language import AssignDotProduct, AssignExpression, AssignNorm, \
    AssignRHS, AssignSolvedRHS, FailStep, If, Raise, ReturnState
from pytools import Record
from pytools.py_codegen import PythonCodeGenerator as PythonEmitter
from pytools.py_codegen import PythonFunctionGenerator as PythonFunctionEmitter
from pymbolic.mapper.stringifier import StringifyMapper
from pymbolic import var
from cStringIO import StringIO

def peek(s):
    """Return a single element from a non-empty iterable."""
    return s.__iter__().next()

class Inst(Record):

    def __init__(self, **kwargs):
        Record.__init__(self, **kwargs)

class BranchInst(Inst):

    def __init__(self, condition, on_true, on_false):
        Inst.__init__(self, condition=condition, on_true=on_true,
                      on_false=on_false)

class AssignInst(Inst):

    def __init__(self, assignment):
        Inst.__init__(self, assignment=assignment)

class JumpInst(Inst):

    def __init__(self, dest):
        Inst.__init__(self, dest=dest)

class ReturnInst(Inst):

    def __init__(self, inst):
        Inst.__init__(self, inst=inst)

class UnreachableInst(Inst):
    
    def __init__(self):
        Inst.__init__(self)
        
class ControlFlowGraph(object):
    """A control flow graph of BasicBlocks."""
    
    def __init__(self, start_block):
        self.start_block = start_block
        self.symbol_table = start_block.symbol_table
        self.update()
        
    def update(self):
        self.number_to_block = {}
        stack = [self.start_block]
        while len(stack) > 0:
            top = stack[-1]
            if top.number in self.number_to_block:
                stack.pop()
                continue
            else:
                self.number_to_block[top.number] = top
                stack.extend(top.successors)
    
    def __iter__(self):
        return self.number_to_block.itervalues()

    def get_dot(self):
        # TODO: Create a graphical represenation.
        pass
    
    def __repr__(self):
        blocks = []
        visited = set()
        stack = [self.start_block]
        while len(stack) > 0:
            top = stack[-1]
            if top.number in visited:
                stack.pop()
                continue
            else:
                blocks += str(top)
                visited.add(top.number)
                stack.extend(top.successors)
        return ''.join(blocks)
    
class BasicBlock(object):
    """A maximal straight-line sequence of instructions."""

    def __init__(self, number, symbol_table):
        self.code = []
        self.predecessors = set()
        self.successors = set()
        self.terminated = False
        self.number = number
        self.symbol_table = symbol_table
    
    def __iter__(self):
        return self.code.__iter__()
    
    def add_unreachable(self):
        assert not self.terminated
        self.code.append(UnreachableInst())
        self.terminated = True
    
    def add_successor(self, succ):
        assert isinstance(succ, BasicBlock)
        self.successors.add(succ)
        succ.predecessors.add(self)
    
    def add_assignment(self, instruction):
        assert not self.terminated
        assert isinstance(instruction, tuple) or \
            isinstance(instruction, AssignExpression) or \
            isinstance(instruction, AssignRHS)
        self.code.append(AssignInst(instruction))
    
    def add_jump(self, dest):
        assert not self.terminated
        self.code.append(JumpInst(dest))
        self.add_successor(dest)
        self.terminated = True
    
    def add_branch(self, cond, on_true, on_false):
        assert not self.terminated
        self.code.append(BranchInst(cond, on_true, on_false))
        self.add_successor(on_true)
        self.add_successor(on_false)
        self.terminated = True
    
    def add_return(self, return_statement):
        assert not self.terminated
        self.code.append(ReturnInst(return_statement))
        self.terminated = True
    
    def __repr__(self):
        o = StringIO()
        mapper = StringifyMapper()
        o.write('===== basic block %s =====\n' % self.number)
        for inst in self.code:
            if isinstance(inst, AssignInst):
                assignment = inst.assignment
                if isinstance(assignment, tuple):
                    o.write('%s <- %s' % (assignment[0], mapper(assignment[1])))
                elif isinstance(assignment, AssignExpression):
                    o.write('%s <- %s' % (assignment.assignee,
                        mapper(assignment.expression)))
                elif isinstance(assignment, AssignRHS):
                    o.write('%s <- %s(%s, %s)' % (assignment.assignees,
                        assignment.component_id, mapper(assignment.t),
                        assignment.rhs_arguments))
            elif isinstance(inst, JumpInst):
                o.write('goto %s' % inst.dest.number)
            elif isinstance(inst, BranchInst):
                o.write('if %s goto %s else %s' % (mapper(inst.condition),
                    inst.on_true.number, inst.on_false.number))
            elif isinstance(inst, ReturnInst):
                if inst.inst:
                    o.write('return (%s, %s)' % (inst.inst.component_id,
                        mapper(inst.inst.expression)))
                else:
                    o.write('return')
            elif isinstance(inst, UnreachableInst):
                o.write('unreachable')
            o.write('\n')
        return o.getvalue()

class FlagAnalysis(object):
    """Determines the values of a set of boolean flags."""
    
    def __init__(self, flags):
        self.all_flags = set(flags)
        self.must_be_true = set()
        self.must_be_false = set()
    
    def set_true(self, flag):
        assert flag in self.all_flags
        new_fa = FlagAnalysis(self.all_flags)
        new_fa.must_be_true.add(flag)
        new_fa.must_be_false.discard(flag)
        return new_fa
    
    def set_false(self, flag):
        assert flag in self.all_flags
        new_fa = FlagAnalysis(self.all_flags)
        new_fa.must_be_false.add(flag)
        new_fa.must_be_true.discard(flag)
        return new_fa
    
    def is_definitely_true(self, flag):
        assert flag in self.all_flags
        return flag in self.must_be_true
    
    def is_definitely_false(self, flag):
        assert flag in self.all_flags
        return flag in self.must_be_false
    
    def __and__(self, other):
        assert isinstance(other, FlagAnalysis)
        assert self.all_flags == other.all_flags
        new_fa = FlagAnalysis(self.all_flags)
        new_fa.must_be_true = self.must_be_true & other.must_be_true
        new_fa.must_be_false = self.must_be_false & other.must_be_false
        return new_fa

class SymbolTable(object):
    """Holds information regarding the variables in a code fragment."""
    
    DAGVariable = 0
    Flag = 1

    def __init__(self):
        self.block_ids = None
        self.variables = {}
        self.named_variables = set()
    
    def __iter__(self):
        return self.variables.__iter__()
    
    def __getitem__(self, var):
        return self.variables[var]

    def add_variable(self, var, ty=None, arg=None):
        # TODO: Use a Record or named tuple
        self.variables[var] = (ty, arg)

    def get_fresh_variable_name(self, prefix=''):
        if prefix not in self.variables and prefix not in self.named_variables:
            self.named_variables.add(prefix)
            return prefix
        suffix = 0
        while prefix + str(suffix) in self.variables:
            suffix += 1
        self.named_variables.add(prefix + str(suffix))
        return prefix + str(suffix)

class SimpleIntGraph(object):
    """Maps a graph-like structure to an adjacency list representation
    of a graph with vertices represented by integers."""

    def __init__(self, vertices, edge_fn):
        # Assign a number to each vertex.
        self.vertex_to_num = {}
        self.num_to_vertex = {}
        num_vertices = 0
        for vertex in vertices:
            if vertex not in self.vertex_to_num:
                self.vertex_to_num[vertex] = num_vertices
                self.num_to_vertex[num_vertices] = vertex
                num_vertices += 1
        self.num_vertices = num_vertices

        self.edges = {}
        # Collect edge information on each vertex.
        for vertex in vertices:
            num = self.vertex_to_num[vertex]
            self.edges[num] = frozenset(map(lambda v: self.vertex_to_num[v],
                                edge_fn(vertex)))

    def __iter__(self):
        return iter(range(0, self.num_vertices))

    def __getitem__(self, num):
        if num not in self.edges:
            raise ValueError('Vertex not found!')
        return self.edges[num]

    def __len__(self):
        return self.num_vertices

    def get_vertex_for_number(self, num):
        if num not in self.num_to_vertex:
            raise ValueError('Vertex not found!')
        return self.num_to_vertex[num]

    def get_number_for_vertex(self, vertex):
        if vertex not in self.vertex_to_num:
            raise ValueError('Vertex not found!')
        return self.vertex_to_num[vertex]

class InstructionDAGIntGraph(SimpleIntGraph):
    """Specialization of SimpleIntGraph that works with instruction DAGs (sets
    of Instructions)."""

    def __init__(self, dag):
        self.id_to_inst = dict((inst.id, inst) for inst in dag)
        self.ids = self.id_to_inst.keys()
        def edge_fn(vertex):
            inst = self.id_to_inst[vertex]
            deps = set(inst.depends_on)
            return deps
        SimpleIntGraph.__init__(self, self.ids, edge_fn)

    def get_number_for_vertex(self, vertex):
        assert False

    def get_vertex_for_number(self, num):
        return self.id_to_inst[self.get_id_for_number(num)]

    def get_id_for_number(self, num):
        return super(InstructionDAGIntGraph, self).get_vertex_for_number(num)

    def get_number_for_id(self, i):
        return super(InstructionDAGIntGraph, self).get_number_for_vertex(i)

class InstructionDAGVerifier(object):
    """Verifies that code is well-formed."""

    def __call__(self, instructions):
        warnings = []
        errors = []
        if not self.verify_all_dependencies_exist(instructions):
            errors += ['Code is missing a dependency']
        if not self.verify_no_circular_dependencies(instructions):
            errors += ['Code has circular dependencies']
        if not self.verify_no_side_effects(instructions):
            warnings += ['Code has side effects']

        return (errors, warnings)

    def verify_returns(self, instructions):
        """Ensures correct usage of return instruction."""
        # TODO
        return True

    def verify_all_dependencies_exist(self, instructions):
        """Ensures that all instruction dependencies exist."""
        # TODO
        return True

    def verify_no_circular_dependencies(self, instructions):
        """Ensures that there are no circular dependencies among the
        instructions."""
        graph = InstructionDAGIntGraph(instructions)
        unvisited = set(graph)
        visiting = set()
        stack = []
        while len(unvisited) > 0:
            stack.append(peek(unvisited))
            while len(stack) > 0:
                top = stack[-1]
                if top in unvisited:
                    unvisited.remove(top)
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

    def verify_no_side_effects(self, instructions):
        """Ensures that instructions contain no side effects."""
        # TODO
        return True

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
        and exit instruction. Every instruction depends on the entry instruction
        while the exit instruction depends on the set instructions_dep_on."""

        ids = set(inst.id for inst in instructions)
        entry_id = self.get_unique_name('entry', ids)
        exit_id = self.get_unique_name('exit', ids)
        import copy
        aug_instructions = copy.deepcopy(instructions)
        aug_deps = set()
        for inst in aug_instructions:
            if inst.id in instructions_dep_on:
                aug_deps.add(inst)
        ent = Entry(id=entry_id)
        ex = Exit(id=exit_id, depends_on=[entry_id] + instructions_dep_on)
        aug_instructions.add(ex)
        for inst in aug_instructions:
            inst.depends_on = frozenset([entry_id] + list(inst.depends_on))
        aug_instructions.add(ent)
        return (aug_instructions, ent, ex)

    def get_unique_name(self, prefix, name_set):
        suffix = 0
        while prefix + str(suffix) in name_set:
            prefix += 1
        return '%s%s' % (prefix, suffix)

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
                top_inst = graph.get_vertex_for_number(top)
                if isinstance(top_inst, If):
                    # If instructions have special dependencies
                    thens = top_inst.then_depends_on
                    stack.extend(map(graph.get_number_for_id, thens))
                    elses = top_inst.else_depends_on
                    stack.extend(map(graph.get_number_for_id, elses))
        return set(map(graph.get_vertex_for_number, reachable))

class InstructionDAGPartitioner(object):
    """Partition a list of instructions into maximal straight line
    sequences with dependency information."""

    def __call__(self, instructions):
        inst_graph = InstructionDAGIntGraph(instructions)
        tr_graph = self.transitive_reduction(inst_graph)
        num_block_graph, num_to_block = self.maximal_blocks(tr_graph)
        id_for_num = inst_graph.get_id_for_number
        to_id = lambda block: tuple(map(id_for_num, block))
        block_graph = dict([to_id(bl), map(to_id, bls)] \
            for (bl, bls) in num_block_graph.iteritems())
        inst_id_to_block = dict([id_for_num(i), to_id(bl)] \
            for (i, bl) in num_to_block.iteritems())

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

    def transitive_reduction(self, dag):
        """Returns a transitive reduction of the input DAG."""
        # 1. Compute u -> v longest paths in the DAG.
        longest_path = dict(((u,v), 0 if u == v else -1) for u in dag
            for v in dag)
        topo_sort = self.topological_sort(dag)
        topo_sort.reverse()
        for i, vertex in enumerate(topo_sort):
            for intermediate_vertex in topo_sort[i:]:
                if longest_path[(vertex, intermediate_vertex)] >= 0:
                    for successor in dag[intermediate_vertex]:
                        old = longest_path[(vertex, successor)]
                        new = 1 + longest_path[(vertex, intermediate_vertex)]
                        longest_path[(vertex, successor)] = max(old, new)

        # 2. Keep only those u -> v edges such that longestPath(u, v) = 1.
        reduction = {}
        for vertex in dag:
            reduction[vertex] = set()
            for successor in dag[vertex]:
                if longest_path[(vertex, successor)] == 1:
                    reduction[vertex].add(successor)
        return reduction

    def maximal_blocks(self, dag):
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
            next = topo_sort.pop()
            if next in visited:
                continue
            visited.add(next)
            block = [next]
            # Traverse down from next.
            while len(dag[next]) == 1:
                next = peek(dag[next])
                if len(dag_inv[next]) == 1:
                    visited.add(next)
                    block.append(next)
                else:
                    break
            block.reverse()
            block = tuple(block)
            for i in block:
                inst_to_block[i] = block
            blocks.add(block)

        # Record the graph structure of the blocks.
        block_graph = {}
        # Get the dependencies of each instruction.
        for block in blocks:
            block_graph[block] = set(inst_to_block[i] for i in dag[block[0]])
        return (block_graph, inst_to_block)

class ControlFlowGraphAssembler(object):
    """Constructs a control-flow graph."""

    def __call__(self, instructions, instructions_dep_on):
        augmenter = InstructionDAGEntryExitAugmenter()
        aug_instructions, ent, ex = \
            augmenter(instructions, instructions_dep_on)
        partitioner = InstructionDAGPartitioner()
        block_graph, inst_id_to_block = partitioner(aug_instructions)
        
        self.block_graph = block_graph
        self.inst_id_to_inst = dict([i.id, i] for i in aug_instructions)
        self.inst_id_to_block = inst_id_to_block
        
        self.initialize_symbol_table(aug_instructions, block_graph)
        self.initialize_flags(block_graph)
        self.basic_block_count = 0
        
        # Find the exit block and create a new basic block out of it.
        entry_block = inst_id_to_block[ent.id]
        exit_block = inst_id_to_block[ex.id]
        
        start_bb = self.get_starting_block()
        landing_pad = self.new_basic_block()
        landing_pad.add_unreachable()
        flag_names = set(self.flags.itervalues())
        flag_analysis = FlagAnalysis(flag_names)
        flag_analysis.must_be_false = set(flag_names)
        top_block, flag_analysis = self.process_block(exit_block, landing_pad,
            flag_analysis)
        start_bb.add_jump(top_block)
        return ControlFlowGraph(start_bb)
    
    def new_basic_block(self):
        name = self.basic_block_count
        self.basic_block_count += 1
        return BasicBlock(name, self.symbol_table)
    
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
            symbol_table.add_variable(flag, ty=SymbolTable.Flag, arg=block)
    
    def initialize_symbol_table(self, aug_instructions, block_graph):
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
        for var in var_names:
            symbol_table.add_variable(var, ty=SymbolTable.DAGVariable)
        
        # Record the RHSs.
        symbol_table.rhs_names = rhs_names
        
        self.symbol_table = symbol_table
    
    def process_block_sequence(self, block_sequence, landing_pad,
        flag_analysis):
        # TODO: Find ways to avoid generating excessive jumps.
        
        if len(block_sequence) == 0:
            return (landing_pad, flag_analysis)
        
        main_bb = self.new_basic_block()
        top_bb, flag_analysis = self.process_block(block_sequence[0],
            main_bb, flag_analysis)
        
        for block in block_sequence[1:]:
            next_bb = self.new_basic_block()
            block_bb, flag_analysis = self.process_block(block, next_bb,
                flag_analysis)
            main_bb.add_jump(block_bb)
            main_bb = next_bb
            
        main_bb.add_jump(landing_pad)
        return (top_bb, flag_analysis)
    
    def get_starting_block(self):
        start_bb = self.new_basic_block()
        # Initialize the flag variables.
        for flag in self.flags.itervalues():
            start_bb.add_assignment((flag, False),)
        return start_bb

    def process_block(self, inst_block, landing_pad, flag_analysis):
        """Produce the control-flow subgraph corresponding to a block of
        instructions."""
        
        get_block_set = lambda inst_set : \
            map(self.inst_id_to_block.__getitem__, inst_set)
        
        # Check the flag analysis to see if we need to compute the block.
        flag = self.flags[inst_block]
        
        main_bb = self.new_basic_block()

        if flag_analysis.is_definitely_true(flag):
            main_bb.add_jump(landing_pad)
            return (main_bb, flag_analysis)
        
        needs_flag = not flag_analysis.is_definitely_false(flag)
                
        # Process all dependencies.
        dependencies = self.block_graph[inst_block]
        top_bb, flag_analysis = self.process_block_sequence(dependencies,
            main_bb, flag_analysis)
        
        if needs_flag:
            # Add code to check and set the flag for the block.
            new_main_bb = self.new_basic_block()
            # Add a jump to the appropriate block from the top block
            from pymbolic.primitives import LogicalNot
            main_bb.add_branch(LogicalNot(var(flag)), new_main_bb, landing_pad)
            # Set the current block being built
            main_bb = new_main_bb
                    
        for instruction_id in inst_block:
            instruction = self.inst_id_to_inst[instruction_id]
            
            if isinstance(instruction, Entry):
                continue
            
            elif isinstance(instruction, Exit):
                main_bb.add_return(None)
                break
                    
            elif isinstance(instruction, If):
                # Get the destination instruction blocks.
                then_blocks = get_block_set(instruction.then_depends_on)
                else_blocks = get_block_set(instruction.else_depends_on)
                merge_bb = self.new_basic_block()
                
                # Emit basic blocks for then and else components.
                then_bb, then_flag_analysis = self.process_block_sequence(
                    then_blocks, merge_bb, flag_analysis)
                else_bb, else_flag_analysis = self.process_block_sequence(
                    else_blocks, merge_bb, flag_analysis)
                
                # Emit branch to then and else blocks.
                main_bb.add_branch(instruction.condition, then_bb, else_bb)
                
                # Set the current basic block to be the merge point.
                flag_analysis = then_flag_analysis & else_flag_analysis
                main_bb = merge_bb
                
            elif isinstance(instruction, ReturnState):
                main_bb.add_return(instruction)
                break
                
            elif isinstance(instruction, AssignExpression) or \
                isinstance(instruction, AssignRHS):
                main_bb.add_assignment(instruction)
        
        if not main_bb.terminated:
            main_bb.add_assignment((flag, True))
            flag_analysis = flag_analysis.set_true(flag)
            main_bb.add_jump(landing_pad)
        return (top_bb, flag_analysis)

class ControlFlowGraphSimplifier:
    """Performs simplification optimizations on the control-flow graph."""
    def __call__(self, control_flow_graph):
        self.merge_basic_blocks(control_flow_graph)
        return control_flow_graph

    def coalesce_jumps(self, control_flow_graph):
        pass

    def remove_dead_basic_blocks(self, control_flow_graph):
        pass

    def merge_basic_blocks(self, control_flow_graph):
        from collections import deque
        queue = deque(block for block in control_flow_graph)
        while len(queue) > 0:
            top = queue.popleft()
            if len(top.successors) == 1:
                succ = peek(top.successors)
                if len(succ.predecessors) == 1:
                    if succ in queue:
                        queue.remove(succ)
                    # Merge the two blocks.
                    top.code = top.code[:-1] + succ.code
                    for succ2 in succ.successors:
                        succ2.predecessors.discard(succ)
                        succ2.predecessors.add(top)
                    top.successors = succ.successors
                    queue.append(top)
                    queue.extend(top.successors)
                    queue.extend(top.predecessors)
        control_flow_graph.update()

class CodeGenerator(object):
    """Base class for code generation."""
    
    def __init__(self, emitter):
        self.emitter = emitter

    def __call__(self, code):
        dag = code.instructions
        self.verify_dag(dag)
        extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        simplifier = ControlFlowGraphSimplifier()

        # Generate initialization code.
        initialization_deps = code.initialization_dep_on
        initialization = extractor(dag, initialization_deps)
        initialization_cfg = assembler(initialization, initialization_deps)
        initialization_cfg = simplifier(initialization_cfg)
        self.emitter.emit_initialization(initialization_cfg)

        # Generate timestepper code.
        stepper_deps = code.step_dep_on
        stepper = extractor(dag, code.step_dep_on)
        stepper_cfg = assembler(stepper, stepper_deps)
        stepper_cfg = simplifier(stepper_cfg)
        self.emitter.emit_stepper(stepper_cfg)
        
        return self.emitter.get_code()

    def verify_dag(self, dag):
        # Verify the dag
        verifier = InstructionDAGVerifier()
        errors, warnings = verifier(dag)
        if warnings:
            for warning in warnings:
                print('Warning: ' + warning)
        if errors:
            for error in errors:
                print('Error:' + error)
            raise Exception
            
class PythonCodeGenerator(CodeGenerator):
    """Converts an instruction DAG to Python code."""

    def __init__(self):
        super(PythonCodeGenerator, self).__init__(self)
        import string
        self.ident_chars = set('_' + string.ascii_letters + string.digits)
        self.class_emitter = PythonClassEmitter('Method')
        self.class_emitter('from leap.vm.exec_numpy import StateComputed, ' +
            'StepCompleted')
        self.finished = False
        self.rhs_map = {}
        self.global_map = {}
    
    def name_global(self, var):
        assert self.is_global(var)
        if var in self.global_map:
            return self.global_map[var]
        elif var == '<t>':
            self.global_map[var] = 'self.t'
        elif var == '<dt>':
            self.global_map[var] = 'self.dt'
        else:
            base = 'global_' + self.filter_variable_name(var)
            self.global_map[var] = self.get_unused_name(base, self.global_map)
        return self.global_map[var]
        
    def is_global(self, var):
        if var == '<t>' or var == '<dt>':
            return True
        elif var.startswith('<state>') or var.startswith('<p>'):
            return True
        else:
            return False
    
    def filter_variable_name(self, name):
        """Converts a variable to a Python identifier."""
        return ''.join(map(lambda c: c if c in self.ident_chars else '_', name))
    
    def name_variables(self, symbol_table):
        """Returns a mapping from variable names to Python identifiers."""
        name_map = {}
        for var in symbol_table:
            if self.is_global(var):
                name_map[var] = self.name_global(var)
                continue
            base = 'v_' + self.filter_variable_name(var)
            name_map[var] = self.get_unused_name(base, name_map)
        return name_map
    
    def get_unused_name(self, base, used_names):
        if base not in used_names:
            return base
        else:
            index = 0
            while base + str(index) in used_names:
                index += 1
            return base + str(index)

    def name_rhss(self, rhss):
        for rhs in rhss:
            if rhs in self.rhs_map:
                continue
            base = 'self.rhs_' + self.filter_variable_name(rhs)
            self.rhs_map[rhs] = self.get_unused_name(base, self.rhs_map)
            
    def get_globals(self, variable_set):
        """Returns the global variables in the given sequence of variable
        names."""
        global_set = set()
        for var in variable_set:
            if var == '<t>' or var == '<dt>':
                global_set.add(var)
            elif var.startswith('<state>') or var.startswith('<p>'):
                global_set.add(var)
        return global_set
    
    def emit_function(self, name, args, control_flow_graph, name_map, rhs_map):
        """Emit the code for a function."""        
        mapper = PythonExpressionMapper(name_map)
        emit = PythonFunctionEmitter(name, args)
        # Emit the control-flow graph as a finite state machine. The state is
        # the current basic block number. Everything is processed in a single
        # outer loop.
        emit('state = 0')
        emit('while True:')
        emit.indent()
        first = True
        for block in control_flow_graph:
            # Emit a single basic block as a sequence of instructions finished
            # by a control transfer.
            emit('%sif state == %d:' % ('el' if not first else '',
                block.number))
            first = False
            emit.indent()
            for inst in block:
                if isinstance(inst, JumpInst):
                    # Jumps transfer state.
                    emit('state = %i' % inst.dest.number)
                    emit('continue')
                    
                elif isinstance(inst, BranchInst):
                    # Branches transfer state.
                    emit('state = %i if (%s) else %i' % (inst.on_true.number,
                        mapper(inst.condition), inst.on_false.number))
                    emit('continue')
                    
                elif isinstance(inst, ReturnInst):
                    inner = inst.inst
                    if not inner:
                        emit('return')
                    else:
                        emit('return (%s, "%s", "%s", %s)' %
                            (mapper(inner.time), inner.time_id,
                            inner.component_id, mapper(inner.expression)))
                    
                elif isinstance(inst, UnreachableInst):
                    # Unreachable instructions should never be executed.
                    emit('raise RuntimeError("Entered an unreachable state!")')
                    
                elif isinstance(inst, AssignInst):
                    assignment = inst.assignment
                    if isinstance(assignment, tuple):
                        var_name = name_map[assignment[0]]
                        expr = mapper(assignment[1])
                        emit('%s = %s' % (var_name, expr))
                    elif isinstance(assignment, AssignExpression):
                        emit('%s = %s' % (name_map[assignment.assignee],
                            mapper(assignment.expression)))
                    elif isinstance(assignment, AssignRHS):
                        assignees = ', '.join(assignment.assignees)
                        rhs = rhs_map[assignment.component_id]
                        time = mapper(assignment.t)
                        arguments = []
                        for arg_pair_list in assignment.rhs_arguments:
                            arg_pair_string = map(arg_pair_list, lambda pair :
                                '%s=%s' % map(mapper, pair))
                            arguments.append(arg_pair_string)
                        calls = []
                        for argument in arguments:
                            calls.append('%s(%s, %s)' % (rhs, time, argument))
                        emit('%s = %s' % (assignees, ', '.join(calls)))
            emit.dedent()
        emit.dedent()
        self.class_emitter.incorporate(emit)
    
    def emit_constructor(self):
        # Save all the rhs components.
        emit = PythonFunctionEmitter('__init__', ('self', 'rhs_map'))
        for rhs in self.rhs_map:
            emit('%s = rhs_map["%s"]' % (self.rhs_map[rhs], rhs))
        emit('return')
        self.class_emitter.incorporate(emit)
    
    def emit_set_up_function(self):
        emit = PythonFunctionEmitter('set_up', ('self', '**kwargs'))
        emit('self.t_start = kwargs["t_start"]')
        emit('self.dt_start = kwargs["dt_start"]')
        emit('state = kwargs["state"]')
        # Save all the state components.
        for state in self.global_map:
            if state == '<t>' or state == '<dt>' or state.startswith('<p>'):
                continue
            emit('%s = state["%s"]' % (self.global_map[state], state))
        emit('return')
        self.class_emitter.incorporate(emit)
    
    def emit_run_function(self):
        emit = PythonFunctionEmitter('run', ('self', '**kwargs'))
        emit('t_end = kwargs["t_end"]')
        emit('last_step = False')
        emit('self.t = self.t_start')
        emit('self.dt = self.dt_start')
        emit('while True:')
        emit.indent()
        emit('if self.t + self.dt >= t_end:')
        emit.indent()
        emit('assert self.t <= t_end')
        emit('self.dt = t_end - self.t')
        emit('last_step = True')
        emit.dedent()
        emit('step = self.step()')
        emit('yield StateComputed(t=step[0], time_id=step[1],' + \
             'component_id=step[2], state_component=step[3])')
        emit('if last_step:')
        emit.indent()
        emit('yield StepCompleted(t=self.t)')
        emit('break')
        emit.dedent()
        self.class_emitter.incorporate(emit)
    
    def emit_initialization(self, control_flow_graph):
        # def initialize(self)
        symbol_table = control_flow_graph.symbol_table
        name_map = self.name_variables(symbol_table)
        self.name_rhss(symbol_table.rhs_names)
        self.emit_function('initialize', ('self',), control_flow_graph,
            name_map, self.rhs_map)
    
    def emit_stepper(self, control_flow_graph):
        # def step(self)
        symbol_table = control_flow_graph.symbol_table
        name_map = self.name_variables(symbol_table)
        self.name_rhss(symbol_table.rhs_names)
        self.emit_function('step', ('self',), control_flow_graph, name_map,
            self.rhs_map)
        
    def get_code(self):
        if not self.finished:
            self.emit_constructor()
            self.emit_set_up_function()
            self.emit_run_function()
            self.finished = True
        return self.class_emitter.get()

class PythonClassEmitter(PythonEmitter):
    """Emits code for a Python class."""
    
    def __init__(self, class_name, superclass='object'):
        super(PythonClassEmitter, self).__init__()
        self.class_name = class_name
        self('class %s(%s):' % (class_name, superclass))
        self.indent()
    
    def incorporate(self, sub_generator):
        """Add the code contained by the subgenerator while respecting the
        current level of indentation."""
        for line in sub_generator.code:
            self(line)
    
class PythonExpressionMapper(StringifyMapper):
    """Converts expressions to Python."""
    
    def __init__(self, variable_names):
        super(PythonExpressionMapper, self).__init__()
        self.variable_names = variable_names
    
    def map_variable(self, expr, enclosing_prec):
        return self.variable_names[expr.name]
