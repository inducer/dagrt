"""Turn timestepper descriptions into source code."""

from __future__ import division, with_statement, print_function

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
from leap.vm.utils import get_variables, peek, is_state_variable, \
    get_unique_name

from pytools import RecordWithoutPickling, DictionaryWithDefault
from pytools import memoize_method
from pytools.py_codegen import PythonCodeGenerator as PythonEmitter
from pytools.py_codegen import PythonFunctionGenerator as PythonFunctionEmitter
from pytools.py_codegen import Indentation

from pymbolic import var
from pymbolic.mapper.stringifier import StringifyMapper

from cgi import escape
from textwrap import TextWrapper


# To do
# - structural analysis
# - code generation for structural analysis
# - write a control flow verifier that checks for correct state handling
# - drawing of control flow graph
# - constant propagation
# - common subexpression elimination


class Inst(RecordWithoutPickling):
    """Base class for instructions in the ControlFlowGraph."""

    def __init__(self, **kwargs):
        if 'block' not in kwargs:
            kwargs['block'] = None
        assert 'terminal' in kwargs
        super(Inst, self).__init__(**kwargs)

    def set_block(self, block):
        self.block = block

    def get_defined_variables(self):
        """Returns the set of variables defined by this instruction."""
        raise NotImplementedError()

    def get_used_variables(self):
        """Returns the set of variables used by this instruction."""
        raise NotImplementedError()

    def get_jump_targets(self):
        """Returns the set of basic blocks that this instruction may jump to."""
        raise NotImplementedError()


class BranchInst(Inst):
    """A conditional branch."""

    def __init__(self, condition, on_true, on_false, block=None):
        super(BranchInst, self).__init__(condition=condition, on_true=on_true,
            on_false=on_false, block=block, terminal=True)

    def get_defined_variables(self):
        return frozenset()

    @memoize_method
    def get_used_variables(self):
        return get_variables(self.condition)

    def get_jump_targets(self):
        return frozenset([self.on_true, self.on_false])

    def __str__(self):
        return 'if %s then goto block %d else goto block %d' % (string_mapper(
            self.condition), self.on_true.number, self.on_false.number)


class AssignInst(Inst):
    """Assigns a value."""

    def __init__(self, assignment, block=None):
        """
        The inner assignment may be in these forms:

            - a (var, expr) tuple, where var is a string and expr is a
                pymbolic expression.

            - One of the following instruction types: AssignExpression,
                AssignRHS, AssignSolvedRHS, AssignDotProduct, AssignNorm
        """
        super(AssignInst, self).__init__(assignment=assignment, block=block,
                                         terminal=False)

    @memoize_method
    def get_defined_variables(self):
        if isinstance(self.assignment, tuple):
            return frozenset([self.assignment[0]])
        else:
            return self.assignment.get_assignees()

    @memoize_method
    def get_used_variables(self):
        if isinstance(self.assignment, tuple):
            return get_variables(self.assignment[1])
        else:
            return self.assignment.get_read_variables()

    def get_jump_targets(self):
        return frozenset()

    def __str__(self):
        assignment = self.assignment
        if isinstance(assignment, tuple):
            return '%s <- %s' % (assignment[0], string_mapper(assignment[1]))
        elif isinstance(assignment, AssignExpression):
            return '%s <- %s' % (assignment.assignee,
                                 string_mapper(assignment.expression))
        elif isinstance(assignment, AssignRHS):
            lines = []
            time = string_mapper(assignment.t)
            rhs = assignment.component_id
            for assignee, arguments in \
                    zip(assignment.assignees, assignment.rhs_arguments):
                arg_list = []
                for arg_pair in arguments:
                    name = arg_pair[0]
                    expr = string_mapper(arg_pair[1])
                    arg_list.append('%s=%s' % (name, expr))
                args = ', '.join(arg_list)
                if len(args) == 0:
                    lines.append('%s <- %s(%s)' % (assignee, rhs, time))
                else:
                    lines.append('%s <- %s(%s, %s)' %
                                 (assignee, rhs, time, args))
            return '\n'.join(lines)


class JumpInst(Inst):
    """Jumps to another basic block."""

    def __init__(self, dest, block=None):
        super(JumpInst, self).__init__(dest=dest, block=block, terminal=True)

    def get_defined_variables(self):
        return frozenset()

    def get_used_variables(self):
        return frozenset()

    def get_jump_targets(self):
        return frozenset([self.dest])

    def __str__(self):
        return 'goto block %d' % self.dest.number


class ReturnInst(Inst):
    """Returns from the function."""

    def __init__(self, expression, block=None):
        super(ReturnInst, self).__init__(expression=expression, block=block,
                                         terminal=True)

    def get_defined_variables(self):
        return frozenset()

    @memoize_method
    def get_used_variables(self):
        return get_variables(self.expression)

    def get_jump_targets(self):
        return frozenset()

    def __str__(self):
        return 'return %s' % string_mapper(self.expression)


class UnreachableInst(Inst):
    """Indicates an unreachable point in the code."""

    def __init__(self, block=None):
        super(UnreachableInst, self).__init__(block=block, terminal=True)

    def get_defined_variables(self):
        return frozenset()

    def get_used_variables(self):
        return frozenset()

    def get_jump_targets(self):
        return frozenset()

    def __str__(self):
        return 'unreachable'


class ControlFlowGraph(object):
    """A control flow graph of BasicBlocks."""

    def __init__(self, start_block):
        self.start_block = start_block
        self.symbol_table = start_block.symbol_table
        self.update()

    def is_acyclic(self):
        """Returns true if the control flow graph contains no loops."""
        return self.acyclic

    def update(self):
        """Traverse the graph and construct the set of basic blocks."""
        self.postorder_traversal = []
        stack = [self.start_block]
        visiting = set()
        visited = set()
        acyclic = True
        while len(stack) > 0:
            top = stack[-1]
            if top not in visited:
                # Mark top as being visited. Add the children of top to the
                # stack.
                visited.add(top)
                visiting.add(top)
                for successor in top.successors:
                    if successor in visiting:
                        acyclic = False
                    elif successor not in visited:
                        stack.append(successor)
            else:
                if top in visiting:
                    # Finished visiting top. Append it to the traversal.
                    visiting.remove(top)
                    self.postorder_traversal.append(top)
                stack.pop()
        self.acyclic = acyclic

    def __iter__(self):
        return self.postorder()

    def __len__(self):
        return len(self.postorder_traversal)

    def postorder(self):
        """Returns an iterator to a postorder traversal of the set of basic
        blocks."""
        return iter(self.postorder_traversal)

    def reverse_postorder(self):
        """Returns an iterator to a reverse postorder traversal of the set of
        basic blocks."""
        return reversed(self.postorder_traversal)

    def get_dot_graph(self):
        """Return a string in the graphviz language that represents the control
        flow graph."""
        # Wrap long lines.
        wrapper = TextWrapper(width=80, subsequent_indent=4 * ' ')

        lines = []
        lines.append('digraph ControlFlowGraph {')

        # Draw the basic blocks.
        for block in self:
            name = str(block.number)

            # Draw the block number.
            lines.append('%s [shape=box,label=<' % name)
            lines.append('<table border="0">')
            lines.append('<tr>')
            lines.append('<td align="center"><font face="Helvetica">')
            lines.append('<b>basic block %s</b>' % name)
            lines.append('</font></td>')
            lines.append('</tr>')

            # Draw the code.
            for inst in block:
                for line in wrapper.wrap(str(inst)):
                    lines.append('<tr>')
                    lines.append('<td align="left">')
                    lines.append('<font face="Courier">')
                    lines.append(escape(line).replace(' ', '&nbsp;'))
                    lines.append('</font>')
                    lines.append('</td>')
                    lines.append('</tr>')
            lines.append('</table>>]')

            # Draw the successor edges.
            for successor in block.successors:
                lines.append('%s -> %d;' % (name, successor.number))

        # Add a dummy entry to mark the start block.
        lines.append('entry [style=invisible];')
        lines.append('entry -> %d;' % self.start_block.number)

        lines.append('}')
        return '\n'.join(lines)

    def __str__(self):
        return '\n'.join(map(str, self.reverse_postorder()))


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
        return iter(self.code)

    def __len__(self):
        return len(self.code)

    def clear(self):
        """Unregisters all instructions."""
        self.delete_instructions(self.code)

    def add_instruction(self, instruction):
        """Appends the given instruction to the code."""
        self.add_instructions([instruction])

    def add_instructions(self, instructions):
        """Appends the given instructions to the code."""
        for instruction in instructions:
            assert not self.terminated
            self.code.append(instruction)
            instruction.set_block(self)
            self.symbol_table.register_instruction(instruction)
            if instruction.terminal:
                # Update the successor information.
                self.terminated = True
                for successor in instruction.get_jump_targets():
                    self.add_successor(successor)

    def delete_instruction(self, instruction):
        """Deletes references to the given instruction."""
        self.delete_instructions([instruction])

    def delete_instructions(self, to_delete):
        """Deletes references to the given instructions."""
        new_code = []
        for inst in self.code:
            if inst not in to_delete:
                new_code.append(inst)
            else:
                self.symbol_table.unregister_instruction(inst)
                if inst.terminal:
                    # If we have removed the terminator, then also update the
                    # successors.
                    self.terminated = False
                    for successor in self.successors:
                        successor.predecessors.remove(self)
                    self.succesors = set()
        self.code = new_code

    def add_unreachable(self):
        """Appends an unreachable instruction to the block."""
        assert not self.terminated
        self.code.append(UnreachableInst(self))
        self.terminated = True

    def add_successor(self, successor):
        """Adds a successor block to the set of successors."""
        assert isinstance(successor, BasicBlock)
        self.successors.add(successor)
        successor.predecessors.add(self)

    def add_assignment(self, instruction):
        """Appends an assignment instruction to the block."""
        assert isinstance(instruction, tuple) or \
            isinstance(instruction, AssignExpression) or \
            isinstance(instruction, AssignRHS)
        self.add_instruction(AssignInst(instruction))

    def add_jump(self, dest):
        """Appends a jump instruction to the block with the given destination.
        """
        self.add_instruction(JumpInst(dest))

    def add_branch(self, condition, on_true, on_false):
        """Appends a branch to the block with the given condition and
        destinations."""
        self.add_instruction(BranchInst(condition, on_true, on_false))

    def add_return(self, expr):
        """Appends a return instruction to the block that returns the given
        expression."""
        self.add_instruction(ReturnInst(expr))

    def __str__(self):
        lines = []
        lines.append('===== basic block %s =====' % self.number)
        lines.extend(str(inst) for inst in self.code)
        return '\n'.join(lines)


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


class SymbolTable(object):
    """Holds information regarding the variables in a code fragment."""

    class SymbolTableEntry(RecordWithoutPickling):
        """Holds information about the contents of a variable. This includes
        all instructions that reference the variable."""

        def __init__(self, *attrs):
            all_attrs = ['is_global', 'is_return_value', 'is_flag']
            attr_dict = dict((attr, attr in attrs) for attr in all_attrs)
            super(SymbolTable.SymbolTableEntry, self).__init__(attr_dict)
            self.references = set()

        @property
        def is_unreferenced(self):
            """Returns whether this variable is not referenced by any
            instructions."""
            return len(self.references) == 0

        def add_reference(self, inst):
            """Add an instruction to the list of instructions that reference
            this variable."""
            self.references.add(inst)

        def remove_reference(self, inst):
            """Remove an instruction from the list of instructions that
            reference this variable."""
            self.references.discard(inst)

    def __init__(self):
        self.block_ids = None
        self.variables = {}
        self.named_variables = set()

    def register_instruction(self, inst):
        variables = inst.get_defined_variables() | inst.get_used_variables()
        for variable in variables:
            assert variable in self.variables
            self.variables[variable].add_reference(inst)

    def unregister_instruction(self, inst):
        variables = inst.get_defined_variables() | inst.get_used_variables()
        for variable in variables:
            assert variable in self.variables
            self.variables[variable].remove_reference(inst)
            if self.variables[variable].is_unreferenced:
                del self.variables[variable]

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, var):
        return self.variables[var]

    def add_variable(self, var, *attrs):
        self.variables[var] = SymbolTable.SymbolTableEntry(*attrs)
        self.named_variables.add(var)

    def get_fresh_variable_name(self, prefix):
        name = get_unique_name(prefix, self.named_variables)
        self.named_variables.add(name)
        return name


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
    of Instructions). Records all the dependency edges in the DAG, including
    conditional dependencies within If statements."""

    def __init__(self, dag):
        self.id_to_inst = dict((inst.id, inst) for inst in dag)
        self.ids = self.id_to_inst.keys()

        def edge_func(vertex):
            inst = self.id_to_inst[vertex]
            deps = set(inst.depends_on)
            if isinstance(inst, If):
                deps |= set(inst.then_depends_on)
                deps |= set(inst.else_depends_on)
            return deps

        super(InstructionDAGIntGraph, self).__init__(self.ids, edge_func)

    def get_unconditional_edges(self, vertex):
        """Return the set of vertices that are adjacent to this vertex by an
        unconditional dependency."""
        inst = self.id_to_inst[self.get_id_for_number(vertex)]
        return set(map(self.get_number_for_id, inst.depends_on))

    def get_conditional_edges(self, vertex):
        """Return the set of vertices that are adjacent to this vertex by a
        conditional dependency (i.e., a branch of an If statement)."""
        inst = self.id_to_inst[self.get_id_for_number(vertex)]
        if isinstance(inst, If):
            deps = inst.then_depends_on + inst.else_depends_on
            return set(map(self.get_number_for_id, deps))
        else:
            return set()

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

    def __call__(self, instructions, *dependency_lists):
        warnings = []
        errors = []

        if not self.verify_instructions_well_typed(instructions):
            errors += ['Instructions are not well formed.']
        elif not self.verify_all_dependencies_exist(instructions,
                                                  *dependency_lists):
            errors += ['Code is missing a dependency.']
        elif not self.verify_no_circular_dependencies(instructions):
            errors += ['Code has circular dependencies.']

        return (errors, warnings)

    def verify_instructions_well_typed(self, instructions):
        """Ensure that all instructions are of the expected format."""
        for inst in instructions:
            # TODO: To what extent should the verifier check the correctness
            # of the input?
            if not isinstance(inst, Instruction):
                return False
        return True

    def verify_all_dependencies_exist(self, instructions, *dependency_lists):
        """Ensures that all instruction dependencies exist."""
        ids = set(inst.id for inst in instructions)
        for inst in instructions:
            deps = set(inst.depends_on)
            if isinstance(inst, If):
                deps |= set(inst.then_depends_on)
                deps |= set(inst.else_depends_on)
            if not deps.issubset(ids):
                return False
        for dependency_list in dependency_lists:
            if not dependency_list.issubset(ids):
                return False
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
            block_graph[block] = set(inst_to_block[i] for i in dag[block[0]]
                if i in original_dag.get_unconditional_edges(block[0]))
        return (block_graph, inst_to_block)


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
        instruction blocks."""

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
        main_bb, flag_analysis = self.process_block_sequence(dependencies,
            top_bb, flag_analysis)

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


class ControlFlowGraphSimplifier(object):
    """Performs simplification optimizations on the control-flow graph."""

    def __call__(self, control_flow_graph):
        changed = False
        changed |= self.coalesce_jumps(control_flow_graph)
        changed |= self.discard_unreachable_blocks(control_flow_graph)
        changed |= self.merge_basic_blocks(control_flow_graph)
        return changed

    def discard_unreachable_blocks(self, control_flow_graph):
        """Searches the control flow graph for reachable blocks by following
        actual edges. Removes all references to blocks that are unreachable."""
        reachable = set()
        stack = [control_flow_graph.start_block]
        while len(stack) > 0:
            top = stack.pop()
            if top not in reachable:
                reachable.add(top)
                stack.extend(top.code[-1].get_jump_targets())
        for block in reachable:
            block.successors &= reachable
            block.predecessors &= reachable
        all_blocks = set([block for block in control_flow_graph])
        changed = reachable != all_blocks
        if changed:
            control_flow_graph.update()
        return changed

    def coalesce_jumps(self, control_flow_graph):
        """Bypasses basic blocks that consist of a single jump instruction."""

        # Find and compute the targets of all blocks that are trivial jumps.
        trivial_jumps = {}
        for block in control_flow_graph.postorder():
            if self.block_is_trivial_jump(block):
                dest = peek(block.successors)
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
                    on_true=terminator.on_true, on_false=terminator.on_false)
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
        """Merges basic blocks that can be trivially combined."""

        regions = []
        has_region = set()
        changed = False

        for block in control_flow_graph.reverse_postorder():

            if block in has_region:
                continue
            region = [block]

            # Extract a maximal basic block.
            while len(block.successors) == 1:
                block = peek(block.successors)
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


class CodeGenerator(object):
    """Base class for code generation."""

    class CodeGenerationError(Exception):

        def __init__(self, errors):
            self.errors = errors

        def __str__(self):
            return 'Errors encountered in input to code generator.\n' + \
                '\n'.join(self.errors)

    def __init__(self, emitter, optimize=True, suppress_warnings=False):
        self.emitter = emitter
        self.suppress_warnings = suppress_warnings
        self.optimize = optimize

    def __call__(self, code):
        dag = code.instructions
        self.verify_dag(dag)
        extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()

        # Generate initialization code.
        initialization_deps = code.initialization_dep_on
        initialization = extractor(dag, initialization_deps)
        initialization_cfg = assembler(initialization, initialization_deps)
        if self.optimize:
            initialization_cfg = optimizer(initialization_cfg)
        self.emitter.emit_initialization(initialization_cfg)

        # Generate timestepper code.
        stepper_deps = code.step_dep_on
        stepper = extractor(dag, code.step_dep_on)
        stepper_cfg = assembler(stepper, stepper_deps)
        if self.optimize:
            stepper_cfg = optimizer(stepper_cfg)
        self.emitter.emit_stepper(stepper_cfg)

        return self.emitter.get_code()

    def verify_dag(self, dag):
        """Verifies that the DAG is well-formed."""
        verifier = InstructionDAGVerifier()
        errors, warnings = verifier(dag)
        if warnings and not self.suppress_warnings:
            from sys import stderr
            for warning in warnings:
                print('Warning: ' + warning, file=stderr)
        if errors:
            raise CodeGenerator.CodeGenerationError(errors)


class PythonCodeGenerator(CodeGenerator):
    """Converts an instruction DAG to Python code."""

    def __init__(self, method_name='Method', **kwargs):
        super(PythonCodeGenerator, self).__init__(self, **kwargs)
        import string
        self.ident_chars = set('_' + string.ascii_letters + string.digits)
        self.class_emitter = PythonClassEmitter(method_name)
        self.class_emitter('from leap.vm.exec_numpy import StateComputed, ' +
            'StepCompleted')
        self.finished = False
        self.rhs_map = {}
        self.global_map = {}

    def name_global(self, var):
        assert is_state_variable(var)
        if var in self.global_map:
            return self.global_map[var]
        elif var == '<t>':
            self.global_map[var] = 'self.t'
        elif var == '<dt>':
            self.global_map[var] = 'self.dt'
        else:
            base = 'self.global' + self.filter_variable_name(var)
            self.global_map[var] = get_unique_name(base, self.global_map)
        return self.global_map[var]

    def filter_variable_name(self, var):
        """Converts a variable to a Python identifier."""
        return ''.join(map(lambda c: c if c in self.ident_chars else '_', var))

    def name_variables(self, symbol_table):
        """Returns a mapping from variable names to Python identifiers."""
        name_map = {}
        for var in symbol_table:
            if is_state_variable(var):
                name_map[var] = self.name_global(var)
                continue
            base = 'v_' + self.filter_variable_name(var)
            name_map[var] = get_unique_name(base, name_map)
        return name_map

    def name_rhss(self, rhss):
        """Returns a mapping from right hand side names to Python identifiers.
        """
        for rhs in rhss:
            if rhs in self.rhs_map:
                continue
            base = 'self.rhs_' + self.filter_variable_name(rhs)
            self.rhs_map[rhs] = get_unique_name(base, self.rhs_map)

    def get_globals(self, variable_set):
        """Returns the global variables in the given sequence of variable
        names."""
        return set(filter(is_state_variable, variable_set))

    def emit_function(self, var, args, control_flow_graph, name_map, rhs_map):
        """Emit the code for a function."""
        mapper = PythonExpressionMapper(name_map)
        emit = PythonFunctionEmitter(var, args)
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
                    emit('return %s' % mapper(inst.expression))

                elif isinstance(inst, UnreachableInst):
                    # Unreachable instructions should never be executed.
                    emit('raise RuntimeError("Entered an unreachable state!")')

                elif isinstance(inst, AssignInst):
                    self.emit_assignment(emit, inst.assignment, name_map,
                        rhs_map, mapper)

            emit.dedent()
        emit.dedent()
        self.class_emitter.incorporate(emit)

    def emit_assignment(self, emit, assignment, name_map, rhs_map, mapper):
        """Generate the Python code for an assignment instruction."""

        if isinstance(assignment, tuple):
            var_name = name_map[assignment[0]]
            expr = mapper(assignment[1])
            emit('%s = %s' % (var_name, expr))
        elif isinstance(assignment, AssignExpression):
            var_name = name_map[assignment.assignee]
            expr = mapper(assignment.expression)
            emit('%s = %s' % (var_name, expr))
        elif isinstance(assignment, AssignRHS):
            # Get the var of the RHS and time
            rhs = rhs_map[assignment.component_id]
            time = mapper(assignment.t)

            # Build list of assignees
            assignees = map(name_map.__getitem__, assignment.assignees)

            # Build up each RHS call
            calls = []
            for argv in assignment.rhs_arguments:
                build_kwarg = lambda pair: '%s=%s' % (pair[0], mapper(pair[1]))
                if len(argv) > 0:
                    argument_string = ', '.join(map(build_kwarg, argv))
                    calls.append('%s(%s, %s)' % (rhs, time, argument_string))
                else:
                    calls.append('%s(%s)' % (rhs, time))

            # Emit the assignment
            for assignee, call in zip(assignees, calls):
                emit('%s = %s' % (assignee, call))

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
        emit('self.t = self.t_start')
        emit('self.dt = self.dt_start')
        emit('state = kwargs["state"]')
        # Save all the state components.
        for state in self.global_map:
            if state == '<t>' or state == '<dt>' or state.startswith('<p>'):
                continue
            elif state.startswith('<state>'):
                emit('%s = state["%s"]' % (self.global_map[state], state[7:]))
        emit('return')
        self.class_emitter.incorporate(emit)

    def emit_run_function(self):
        emit = PythonFunctionEmitter('run', ('self', '**kwargs'))
        emit('t_end = kwargs["t_end"]')
        emit('last_step = False')
        emit('while True:')
        with Indentation(emit):
            emit('if self.t + self.dt >= t_end:')
            with Indentation(emit):
                emit('assert self.t <= t_end')
                emit('self.dt = t_end - self.t')
                emit('last_step = True')
            emit('step = self.step()')
            emit('yield StateComputed(t=step[0], time_id=step[1], ' +
                'component_id=step[2], state_component=step[3])')
            emit('if last_step:')
            with Indentation(emit):
                emit('yield StepCompleted(t=self.t)')
                emit('break')
        self.class_emitter.incorporate(emit)

    def emit_initialization(self, control_flow_graph):
        symbol_table = control_flow_graph.symbol_table
        name_map = self.name_variables(symbol_table)
        self.name_rhss(symbol_table.rhs_names)
        self.emit_function('initialize', ('self',), control_flow_graph,
            name_map, self.rhs_map)

    def emit_stepper(self, control_flow_graph):
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
    """Converts expressions to Python code."""

    def __init__(self, variable_names):
        super(PythonExpressionMapper, self).__init__(repr)
        self.variable_names = variable_names

    def map_foreign(self, expr, *args):
        if expr is None:
            return self.map_none(expr)
        elif isinstance(expr, str):
            return self.map_string(expr)
        else:
            return super(PythonExpressionMapper, self).map_foreign(expr, *args)

    def map_string(self, expr):
        return repr(expr)

    def map_variable(self, expr, enclosing_prec):
        return self.variable_names[expr.name]

    def map_none(self, expr):
        return 'None'


string_mapper = PythonExpressionMapper(DictionaryWithDefault(lambda x: x))


class CCodeGenerator(CodeGenerator):
    """struct State { ... }"""
    """State run_time_stepper(State);"""
    """State initialize_time_stepper(StateTy state);"""

    def __init__(self):
        pass


class StructuredControlFlowGraph(object):
    pass


class StructuredControlFlowExtractor(object):
    pass


class Optimizer(object):
    """Performs optimizations on the code in a control flow graph."""

    def __call__(self, control_flow_graph):
        cfg_simplify = ControlFlowGraphSimplifier()
        adce = AggressiveDeadCodeElimination()

        iterations = 0
        changed = True
        while changed and iterations < 5:
            # Attempt to iterate until convergence.
            changed = False
            changed |= cfg_simplify(control_flow_graph)
            changed |= adce(control_flow_graph)
            iterations += 1

        return control_flow_graph


class AggressiveDeadCodeElimination(object):
    """Removes dead code."""

    def __call__(self, control_flow_graph):
        reaching_definitions = ReachingDefinitions(control_flow_graph)

        # Find all trivially essential instructions.
        essential = set()
        for block in control_flow_graph:
            essential |= \
                set(inst for inst in block if self.is_trivially_essential(inst))

        # Working backwards from the set of trivially essential instructions,
        # discover all essential instructions.
        worklist = list(essential)
        while len(worklist) > 0:
            inst = worklist.pop()
            dependencies = self.get_dependent_instructions(inst,
                reaching_definitions)
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
        insts = set(pair[1] for pair in definitions if pair[0] in variables)
        return insts

    def is_trivially_essential(self, inst):
        if isinstance(inst, ReturnInst):
            # All return instructions are essential.
            return True
        elif isinstance(inst, BranchInst) or isinstance(inst, JumpInst):
            # All control flow instructions are essential. This is a pessimistic
            # assumption and may be improved upon if the worklist algorithm used
            # control dependence to discover the set of essential control flow
            # instructions.
            return True
        elif isinstance(inst, AssignInst):
            symbol_table = inst.block.symbol_table
            # All assignments to state variables are essential.
            for assignee in inst.get_defined_variables():
                if symbol_table[assignee].is_global:
                    return True
        else:
            return False


class ReachingDefinitions(object):
    """Performs a reaching definitions analysis and computes use-def chains."""

    def __init__(self, control_flow_graph):
        # A definition is a pair (variable, instruction) representing a variable
        # name and the instruction which defines the variable.

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
                changed = changed or len(reach) > len(def_in[block])
                def_in[block] = reach

                kill = def_kill[block]
                reach_out = self.remove_killed(reach, kill)

                reach_out |= def_gen[block]
                changed = changed or len(reach_out) > len(def_out[block])

                def_out[block] = reach_out

            # If the graph is acyclic then only a single iteration is required.
            if control_flow_graph.is_acyclic():
                break

        self.def_in = def_in
        self.def_out = def_out

    def get_gen_and_kill_sets(self, block, point):
        """Returns the gen and kill sets."""
        last_def = {}
        for pos, inst in enumerate(block.code):
            if pos == point:
                break
            for name in inst.get_defined_variables():
                last_def[name] = inst
        return (set(last_def.iteritems()), set(last_def.iterkeys()))

    def remove_killed(self, definitions, kill):
        """Returns the result of removing all definitions that are killed."""
        return set(pair for pair in definitions if pair[0] not in kill)

    @memoize_method
    def get_reaching_definitions(self, instruction):
        """Returns the set of all definitions that reach the instruction on some
        execution path."""
        block = instruction.block
        index = block.code.index(instruction)
        gen, kill = self.get_gen_and_kill_sets(block, index)
        return gen | self.remove_killed(self.def_in[block], kill)
