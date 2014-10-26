"""A flow-graph based intermediate representation"""

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

from pytools import RecordWithoutPickling, memoize_method
from leap.vm.utils import get_unique_name, get_variables, TODO
from leap.vm.language import AssignExpression, AssignRHS
from pymbolic.mapper.stringifier import StringifyMapper
from textwrap import TextWrapper
from cgi import escape


string_mapper = StringifyMapper()


# {{{ dag instructions

class Inst(RecordWithoutPickling):
    """Base class for instructions in the control flow graph.

    .. attribute:: block
        The containing BasicBlock (may be None)

    .. attribute:: terminal
        A boolean valued class attribute. Indicates if the instances of the
        instruction class function as the terminating instruction of a basic
        block.
    """

    terminal = False

    def __init__(self, **kwargs):
        if 'block' not in kwargs:
            kwargs['block'] = None
        super(Inst, self).__init__(**kwargs)

    def set_block(self, block):
        self.block = block

    def get_defined_variables(self):
        """Return the set of variables defined by this instruction."""
        raise NotImplementedError()

    def get_used_variables(self):
        """Return the set of variables used by this instruction."""
        raise NotImplementedError()

    def get_jump_targets(self):
        """Return the set of basic blocks that this instruction may jump to.
        """
        raise NotImplementedError()


class AssignInst(Inst):
    """Assigns a value.

    .. attribute:: assignment

        The assignment expression may be in these forms:
            - a (`var`, `expr`) tuple, where `var` is a string and `expr` is a
                pymbolic expression.
            - One of the following :class:`leap.vm.language.Instruction` types:
                - AssignExpression
                - AssignRHS
                - AssignSolvedRHS
                - AssignDotProduct
                - AssignNorm
    """

    def __init__(self, assignment, block=None):
        super(AssignInst, self).__init__(assignment=assignment, block=block)

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
            return '{name} <- {expr}'.format(name=assignment[0],
                                             expr=string_mapper(assignment[1]))
        if isinstance(assignment, AssignExpression):
            return '{name} <- {expr}'.format(name=assignment.assignee,
                expr=string_mapper(assignment.expression))
        if isinstance(assignment, AssignRHS):
            lines = []
            time = string_mapper(assignment.t)
            rhs = assignment.component_id
            for assignee, arguments in \
                    zip(assignment.assignees, assignment.rhs_arguments):
                arg_list = []
                for arg_pair in arguments:
                    name = arg_pair[0]
                    expr = string_mapper(arg_pair[1])
                    arg_list.append(name + '=' + expr)
                args = ', '.join(arg_list)
                if len(args) == 0:
                    lines.append('{name} <- {rhs}({t})'.format(name=assignee,
                        rhs=rhs, t=time))
                else:
                    lines.append('{name} <- {rhs}({t}, {args})'.format(
                                name=assignee, rhs=rhs, t=time, args=args))
            return '\n'.join(lines)
        raise TODO('Implement string representation for all assignment types')


class YieldInst(Inst):
    """Generates a value.

    .. attribute:: expression
       A pymbolic expression for the generated value.
    """

    def __init__(self, expression, block=None):
        super(YieldInst, self).__init__(expression=expression, block=block)

    def get_defined_variables(self):
        return frozenset()

    @memoize_method
    def get_used_variables(self):
        return get_variables(self.expression)

    def get_jump_targets(self):
        return frozenset()

    def __str__(self):
        return 'yield ' + string_mapper(self.expression)


class TerminatorInst(Inst):
    """Base class for instructions that terminate a basic block."""
    terminal = True


class BranchInst(TerminatorInst):
    """A conditional branch.

    .. attribute:: condition
        A pymbolic expression for the condition

    .. attribute:: on_true
    .. attribute:: on_false
        The BasicBlocks that are destinations of the branch
    """

    def __init__(self, condition, on_true, on_false, block=None):
        super(BranchInst, self).__init__(condition=condition, on_true=on_true,
                                         on_false=on_false, block=block)

    def get_defined_variables(self):
        return frozenset()

    @memoize_method
    def get_used_variables(self):
        return get_variables(self.condition)

    def get_jump_targets(self):
        return frozenset([self.on_true, self.on_false])

    def __str__(self):
        return \
            'if {cond} then goto block {true} else goto block {false}'.format(
                cond=string_mapper(self.condition), true=self.on_true.number,
                false=self.on_false.number)


class JumpInst(TerminatorInst):
    """Jumps to another basic block.

    .. attribute:: dest
        The BasicBlock that is the destination of the jump
    """

    def __init__(self, dest, block=None):
        super(JumpInst, self).__init__(dest=dest, block=block)

    def get_defined_variables(self):
        return frozenset()

    def get_used_variables(self):
        return frozenset()

    def get_jump_targets(self):
        return frozenset([self.dest])

    def __str__(self):
        return 'goto block ' + str(self.dest.number)


class ReturnInst(TerminatorInst):
    """Returns from the function."""

    def __init__(self, block=None):
        super(ReturnInst, self).__init__(block=block)

    def get_defined_variables(self):
        return frozenset()

    def get_used_variables(self):
        return frozenset()

    def get_jump_targets(self):
        return frozenset()

    def __str__(self):
        return 'return'


# }}}


# {{{ basic block

class BasicBlock(object):
    """A maximal straight-line sequence of instructions.

    .. attribute:: code
        A list of Insts in the basic block

    .. attribute:: number
        The basic block number

    .. attribute:: predecessors
    .. attribute:: successors
        The sets of predecessor and successor blocks

    .. attribute:: terminated
        A boolean indicating if the code ends in a TerminatorInst
    """

    def __init__(self, number, function):
        self.code = []
        self.predecessors = set()
        self.successors = set()
        self.terminated = False
        self.number = number
        self._function = function

    def __iter__(self):
        return iter(self.code)

    def __len__(self):
        return len(self.code)

    @property
    def symbol_table(self):
        """Return the symbol table."""
        return self._function.symbol_table

    def clear(self):
        """Unregister all instructions."""
        self.delete_instructions(self.code)

    def add_instruction(self, instruction):
        """Append the given instruction to the code."""
        self.add_instructions([instruction])

    def add_instructions(self, instructions):
        """Append the given instructions to the code."""
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
        """Delete references to the given instruction."""
        self.delete_instructions([instruction])

    def delete_instructions(self, to_delete):
        """Delete references to the given instructions."""
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

    def add_successor(self, successor):
        """Add a successor block to the set of successors."""
        assert isinstance(successor, BasicBlock)
        self.successors.add(successor)
        successor.predecessors.add(self)

    def add_assignment(self, instruction):
        """Append an assignment instruction to the block."""
        self.add_instruction(AssignInst(instruction))

    def add_jump(self, dest):
        """Append a jump instruction to the block with the given destination.
        """
        self.add_instruction(JumpInst(dest))

    def add_branch(self, condition, on_true, on_false):
        """Append a branch to the block with the given condition and
        destinations.
        """
        self.add_instruction(BranchInst(condition, on_true, on_false))

    def add_return(self):
        """Append a return instruction to the block."""
        self.add_instruction(ReturnInst())

    def add_yield(self, expr):
        """Append a yield instruction to the block with the given
        expression."""
        self.add_instruction(YieldInst(expr))

    def __str__(self):
        lines = []
        lines.append('===== basic block {num} ====='.format(num=self.number))
        lines.extend(str(inst) for inst in self.code)
        return '\n'.join(lines)

# }}}


# {{{ function

class Function(object):
    """A control flow graph of BasicBlocks.

    .. attribute:: name
        The name of the function

    .. attribute:: entry_block
        The BasicBlock that is the entry of the flow graph

    .. attribute:: symbol_table
        The associated SymbolTable
    """

    def __init__(self, name, symbol_table):
        self.name = name
        self.symbol_table = symbol_table

    def assign_entry_block(self, block):
        self.entry_block = block
        self.update()

    def is_acyclic(self):
        """Return true if the control flow graph contains no loops."""
        return self.acyclic

    def update(self):
        """Traverse the graph and construct the set of basic blocks."""
        self.postorder_traversal = []
        stack = [self.entry_block]
        visiting = set()
        visited = set()
        acyclic = True
        while stack:
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
        """Return an iterator to a postorder traversal of the set of basic
        blocks.
        """
        return iter(self.postorder_traversal)

    def reverse_postorder(self):
        """Return an iterator to a reverse postorder traversal of the set of
        basic blocks.
        """
        return reversed(self.postorder_traversal)

    def get_dot_graph(self):
        """Return a string in the graphviz language that represents the
        control flow graph.
        """
        # Wrap long lines.
        wrapper = TextWrapper(width=80, subsequent_indent=4 * ' ')

        lines = []
        lines.append('digraph ControlFlowGraph {')

        # Draw the basic blocks.
        for block in self:
            name = str(block.number)

            # Draw the block number.
            lines.append('{name} [shape=box,label=<'.format(name=name))
            lines.append('<table border="0">')
            lines.append('<tr>')
            lines.append('<td align="center"><font face="Helvetica">')
            lines.append('<b>basic block {name}</b>'.format(name=name))
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
                lines.append('{name} -> {succ};'.format(name=name,
                                                        succ=successor.number))

        # Add a dummy entry to mark the start block.
        lines.append('entry [style=invisible];')
        lines.append('entry -> {entry};'.format(entry=self.entry_block.number))

        lines.append('}')
        return '\n'.join(lines)

    def __str__(self):
        return '\n'.join(map(str, self.reverse_postorder()))

# }}}


# {{{ symbol table

class SymbolTableEntry(RecordWithoutPickling):
    """Holds information about the contents of a variable. This includes
    all instructions that reference the variable.

    .. attribute:: is_return_value
    .. attribute:: is_flag
    .. attribute:: is_global

    .. attribute:: references

        A list of :class:`Inst` instances referencing this variable.
    """

    def __init__(self,
            is_return_value=False,
            is_flag=False,
            is_global=False,
            references=None):
        if references is None:
            references = set()

        super(SymbolTableEntry, self).__init__(
                is_return_value=is_return_value,
                is_flag=is_flag,
                is_global=is_global,
                references=references,
                )

    @property
    def is_unreferenced(self):
        """Return whether this variable is not referenced by any
        instructions.
        """
        return len(self.references) == 0

    def add_reference(self, inst):
        """Add an instruction to the list of instructions that reference this
        variable.
        """
        self.references.add(inst)

    def remove_reference(self, inst):
        """Remove an instruction from the list of instructions that reference
        this variable.
        """
        self.references.discard(inst)


class SymbolTable(object):
    """Holds information regarding the variables in a code fragment.

    .. attribute:: variables
        A map from variable name to SymbolTableEntry
    """

    def __init__(self):
        self.variables = {}
        self._named_variables = set()

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

            # FIXME This is weird--we shouldn't automatically delete
            # unreferenced variables. There's a funny asymmetry
            # between freshly created variables that don't have
            # a reference yet and ones that lose their last reference
            # and thus get deleted.
            if self.variables[variable].is_unreferenced:
                del self.variables[variable]

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, var):
        return self.variables[var]

    def add_variable(self, var, **kwargs):
        self.variables[var] = SymbolTableEntry(**kwargs)
        self._named_variables.add(var)

    def get_fresh_variable_name(self, prefix):
        name = get_unique_name(prefix, self._named_variables)
        self._named_variables.add(name)
        return name

# }}}

# vim: foldmethod=marker
