"""Creation of a control tree from intermediate code"""

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

from .ir import BranchInst
from .structured_ir import SingleNode, BlockNode, IfThenNode, IfThenElseNode, \
    UnstructuredIntervalNode
from pytools import one, DictionaryWithDefault


def distinct(*items):
    """Return True if and only if each item in the list is a distinct object."""
    for i in range(0, len(items)):
        for j in range(i + 1, len(items)):
            if items[i] is items[j]:
                return False
    return True


# {{{ node check functions

def _check_for_if_then_node(node):
    if len(node.successors) != 2:
        return None

    # Get then and merge nodes.
    then_node, merge_node = tuple(node.successors)

    # Swap then and merge nodes if necessary.
    if then_node in merge_node.successors:
        then_node, merge_node = merge_node, then_node

    # Check for expected structure.
    if len(then_node.successors) != 1 or \
            len(then_node.predecessors) != 1 or \
            merge_node not in then_node.successors or \
            not distinct(node, then_node, merge_node):
        return None
    return IfThenNode(node, then_node)


def _check_for_if_then_else_node(node):
    if len(node.successors) != 2:
        return None

    # Get then and else nodes.
    branch = node.exit_block.code[-1]
    assert isinstance(branch, BranchInst)

    then_basic_block = branch.on_true
    else_basic_block = branch.on_false
    then_node, else_node = tuple(node.successors)
    if else_node.entry_block is then_basic_block:
        then_node, else_node = else_node, then_node

    assert then_node.entry_block is then_basic_block
    assert else_node.entry_block is else_basic_block

    # Check for no other predecessors to then and else.
    if len(then_node.predecessors) != 1 or \
            len(else_node.predecessors) != 1:
        return None

    # Check for a common merge point of then and else.
    merge_node = None
    successors = then_node.successors | else_node.successors
    if len(successors) > 1:
        return None
    if successors:
        merge_node = one(successors)

    # Check for block distinctness.
    if not distinct(node, then_node, else_node, merge_node):
        return None
    return IfThenElseNode(node, then_node, else_node)


def _check_for_block_node(node):
    # Follow the predecessors.
    predecessor_nodes = []
    current_node = node
    while len(current_node.predecessors) == 1 and \
          len(one(current_node.predecessors).successors) == 1:
        current_node = one(current_node.predecessors)
        predecessor_nodes.append(current_node)

    # Follow the successors.
    successor_nodes = []
    current_node = node
    while len(current_node.successors) == 1 and \
          len(one(current_node.successors).predecessors) == 1:
        current_node = one(current_node.successors)
        successor_nodes.append(current_node)

    # Check if a sequence has been detected.
    if not predecessor_nodes and not successor_nodes:
        return None

    # Check if the sequence is single-exit.
    last_node = successor_nodes[-1] if successor_nodes else node
    if not hasattr(last_node, "exit_block"):
        return None

    # Construct the block.
    return BlockNode(list(reversed(predecessor_nodes)) + [node] +
                     successor_nodes)

# }}}


class StructuralExtractor(object):
    """Top-level entry point to create a control tree from a control flow graph.

    Based on:
       Sharir, Micha. "Structural analysis: a new approach to flow analysis
        in optimizing compilers." Computer Languages 5.3 (1980): 141-153.
    """

    def __call__(self, function):
        """
        :arg function: a :class:`leap.vm.codegen.ir.Function` instance
        """

        # Wrap all basic blocks with SingleNodes.
        block_nodes = dict((block, SingleNode(block)) for block in function)
        # Add successors / predecessors to block nodes.
        for block, node in block_nodes.items():
            node.successors |= {block_nodes[succ] for succ in block.successors}
            node.predecessors |= {block_nodes[pred] for pred in
                                  block.predecessors}

        nodes = list(reversed([block_nodes[node] for node in
                               function.postorder()]))

        while len(nodes) > 1:
            # Map from node to its containing structure.
            struct_of = DictionaryWithDefault(lambda x: x)

            changed = False

            for node in nodes:
                # Skip this node if it has already been made part of a larger
                # node this iteration.
                if struct_of[node] is not node:
                    continue

                # Check for the structural type of the node.
                new_node = _check_for_block_node(node)
                new_node = new_node or _check_for_if_then_node(node)
                new_node = new_node or _check_for_if_then_else_node(node)

                if not new_node:
                    continue

                changed = True
                # Update struct_of if a new node has been built.
                for inner_node in new_node.nodes:
                    struct_of[inner_node] = new_node

            # If there are no updates, create an unstructured interval object.
            if not changed:
                return UnstructuredIntervalNode(nodes)

            # Otherwise, the set of nodes has been updated.  Build a new list
            # of nodes for the next iteration that preserves the reverse
            # postorder ordering of the nodes.
            new_nodes = []

            for node in nodes:
                # Find the outermost containing structure of the old node.
                outermost_node = struct_of[node]
                while struct_of[outermost_node] is not outermost_node:
                    outermost_node = struct_of[outermost_node]
                struct_of[node] = outermost_node

                # Append the outermost structure if it shares the same entry
                # block as the old node.
                if outermost_node.entry_block is node.entry_block:
                    new_nodes.append(outermost_node)

            nodes = new_nodes

        return one(nodes)

# vim: foldmethod=marker
