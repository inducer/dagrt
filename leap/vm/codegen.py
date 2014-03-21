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
from pytools import Record

def peek(s):
    """Return a single element from a non-empty iterable."""
    return s.__iter__().next()

class CFG(object):

    def get_dot(self):
        pass

class FlagAnalysis(object):
    # Objects
    # defs - the variables defined by this basic block
    # all_def_in - the variables defined on all paths into this block
    # none_def_in - the variables not defined on any paths into this block
    # all_def_out - the variables defined on all paths out of this block
    # none_def_out - the variables not defined on any paths out of this block
    pass

class BasicBlock(object):

    def __init__(self):
        self.flag_analysis = FlagAnalysis()

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

    def __init__(self, expr):
        Inst.__init__(self, expr=expr)

class BasicBlockBuilder(object):
    pass

class SymbolTable(object):

    def __init__(self):
        pass

    def get_variable_info(self, var):
        pass

    def get_fresh_variable_name(self, prefix):
        pass

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

class InstructionDAGGraph(SimpleIntGraph):
    """Specialization of SimpleIntGraph that works with instruction DAGs (sets
    of Instructions)."""

    def __init__(self, dag):
        self.id_to_inst = dict((inst.id, inst) for inst in dag)
        self.ids = self.id_to_inst.keys()
        SimpleIntGraph.__init__(self, self.ids,
                                lambda i : self.id_to_inst[i].depends_on)

    def get_number_for_vertex(self, vertex):
        assert False

    def get_vertex_for_number(self, num):
        return self.id_to_inst[self.get_id_for_number(num)]

    def get_id_for_number(self, num):
        return super(InstructionDAGGraph, self).get_vertex_for_number(num)

    def get_number_for_id(self, id):
        return super(InstructionDAGGraph, self).get_number_for_vertex(id)

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
        return True

    def verify_all_dependencies_exist(self, instructions):
        """Ensures that all instruction dependencies exist."""
        return True

    def verify_no_circular_dependencies(self, instructions):
        """Ensures that there are no circular dependencies among the
        instructions."""
        graph = InstructionDAGGraph(instructions)
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
        return True

class Entry(Instruction):

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
        entry = Entry(id=entry_id)
        exit = Exit(id=exit_id, depends_on=[entry_id] + instructions_dep_on)
        aug_instructions.add(exit)
        for inst in aug_instructions:
            inst.depends_on = frozenset([entry_id] + list(inst.depends_on))
        aug_instructions.add(entry)
        return (aug_instructions, entry, exit)

    def get_unique_name(self, prefix, name_set):
        suffix = 0
        while prefix + str(suffix) in name_set:
            prefix += 1
        return '%s%s' % (prefix, suffix)

class InstructionDAGExtractor(object):

    def __call__(self, dag, dependencies):
        graph = InstructionDAGGraph(dag)
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
        inst_graph = InstructionDAGGraph(instructions)
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
        longest_path = dict(
            ((u,v), 0 if u == v else -1) for u in dag for v in dag)
        ts = self.topological_sort(dag)
        ts.reverse()
        for i, u in enumerate(ts):
            for v in ts[i:]:
                if longest_path[(u, v)] >= 0:
                    for vv in dag[v]:
                        longest_path[(u, vv)] = max(longest_path[(u, vv)],
                            1 + longest_path[(u, v)])

        # 2. Keep only those u -> v edges such that longestPath(u, v) = 1.
        tr = {}
        for u in dag:
            tr[u] = set()
            for v in dag[u]:
                if longest_path[(u, v)] == 1:
                    tr[u].add(v)
        return tr

    def maximal_blocks(self, dag):
        """Returns a partition of the DAG into maximal blocks of straight-line
        pieces."""
        # Compute the inverse of the DAG.
        dag_inv = dict((u, set()) for u in dag)
        for u, v in dag.iteritems():
            for v_elt in v:
                dag_inv[v_elt].add(u)

        # Traverse the DAG extracting maximal straight line sequences into
        # blocks.
        ts = self.topological_sort(dag)
        visited = set()
        blocks = set()
        inst_to_block = {}
        while len(ts) > 0:
            next = ts.pop()
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
            for b in block:
                inst_to_block[b] = block
            blocks.add(block)

        # Record the graph structure of the blocks.
        block_graph = {}
        for block in blocks:
            block_graph[block] = set(inst_to_block[i] for i in dag[block[0]])
        return (block_graph, inst_to_block)

class CFGAssembler(object):
    """Constructs a control-flow graph."""

    def __call__(self, instructions, instructions_dep_on):
        augmenter = InstructionDAGEntryExitAugmenter()
        aug_instructions, entry, exit = \
            augmenter(instructions, instructions_dep_on)
        partitioner = InstructionDAGPartitioner()
        block_graph, inst_id_to_block = partitioner(aug_instructions)

    def init_symbol_table(self, block_graph):
        symbol_table = SymbolTable()

        # Get a list of all used variable names.
        var_names = set()
        get_names = lambda var_set: map(lambda var: var.name, var_set)
        for inst in aug_instructions:
            var_names |= get_names(inst.get_assignees())
            var_names |= get_names(inst.get_read_variables())

        # Create an ID for each block.
        block_id_graph = SimpleIntGraph(block_graph, lambda bl: block_graph[bl])

        # Create a flag for each block and insert into the symbol table.
        for block_id in block_id_graph:
            pass

    def process_block(self, inst_block, jump_dest):
        """Produce the control-flow subgraph corresponding to a block of
        instructions."""
        """
        landingpad = new_basic_block
        for dependency in block_dependencies:
            process_block(dependency, landingpad)
        for instruction in inst_block:
            case instruction of ==>
                Entry => emit initialization
                Exit => do nothing
                Return => perform return
                """

class CodeGenerator(object):

    def __call__(self, code, emitter):
        dag = code.instructions
        self.verify_dag(dag)
        extractor = InstructionDAGExtractor()
        assembler = CFGAssembler()

        initialization_deps = code.initialization_dep_on
        initialization = extractor(dag, initialization_deps)
        initialization_cfg = assembler(initialization, initialization_deps)
        emitter.emit_initialization(initialization_cfg)

        stepper_deps = code.step_dep_on
        stepper = extractor(dag, code.step_dep_on)
        stepper_cfg = assembler(stepper, stepper_deps)
        emitter.emit_stepper(stepper_cfg)

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

class PythonEmitter(object):

    def __init__(self, output=None):
        pass

    def normalize_cfg(self, cfg):
        pass

    def emit_initialization(self, cfg):
        pass

    def emit_stepper(self, cfg):
        pass

# Code generation steps
# 1. Add dummy entry instruction.
# 2. Perform transitive reduction.
# 3. Partition into maximal straight line sequences.
# 4. Assemble CFG.
# 5. SSA form?
# 6. PRE + DCE + Const. prop.
