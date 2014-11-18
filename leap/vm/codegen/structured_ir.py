"""A tree-based intermediate representation for structured code"""

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

from pytools import RecordWithoutPickling


class ControlNode(RecordWithoutPickling):
    """Represents an item in the control tree.

    Attributes:
     - nodes: the set of nodes contained by this control node
     - successors: the set of successor nodes
     - predecessors: the set of predecessor nodes
    """

    def __init__(self, **kwargs):
        assert 'successors' not in kwargs
        assert 'predecessors' not in kwargs
        kwargs.update({'successors': set(), 'predecessors': set()})
        super(ControlNode, self).__init__(**kwargs)

    # The update_* methods are used to inform the predecessor /
    # successor nodes when the current node is updated.

    def update_predecessors(self, old_successor, exclude=frozenset()):
        """Replace instances of old_successors with self in the
        predecessors' sets of successors.
        """
        for predecessor in self.predecessors - exclude:
            predecessor.successors -= {old_successor}
            predecessor.successors.add(self)

    def update_successors(self, old_predecessor, exclude=frozenset()):
        """Replace instances of old_predecessor with self in the
        successors' sets of predecessors.
        """
        for successor in self.successors - exclude:
            successor.predecessors -= {old_predecessor}
            successor.predecessors.add(self)

    # RecordWithoutPickling.__repr__() is really slow on control trees.
    __repr__ = object.__repr__


class SingleNode(ControlNode):
    """Represents a single basic block.

    Attributes:
     - basic_block: the basic block
    """

    def __init__(self, basic_block):
        """Note: Does not set the successors and predecessors."""
        super(SingleNode, self).__init__(nodes={basic_block},
                                         basic_block=basic_block,
                                         entry_block=basic_block,
                                         exit_block=basic_block)


class BlockNode(ControlNode):
    """Represents a straight line sequence of control nodes.

    Nodes B_1, B_2, ..., B_n form a block if B_2, B_3, ..., B_{n-1} are
    single entry single exit nodes with successor(B_i) = predecessor(B_{i+1}).

    Attributes:
     - node_list: the list of nodes in the block
    """

    def __init__(self, node_list):
        if not node_list:
            raise ValueError('empty node list')
        super(BlockNode, self).__init__(nodes=set(node_list),
                                        node_list=list(node_list),
                                        entry_block=node_list[0].entry_block,
                                        exit_block=node_list[-1].exit_block)

        self.predecessors |= node_list[0].predecessors
        self.update_predecessors(node_list[0])

        self.successors |= node_list[-1].successors
        self.update_successors(node_list[-1])


class IfThenNode(ControlNode):
    """Represents an if-then control structure.

    Distinct nodes A and B form an if-then control structure if B is
    single-entry and there exists another node C distinct from A and B
    such that:

      (i)  A has exactly two successors B and C
      (ii) B has exactly one successor C

    Note that C is not itself part of the control structure.

    Attributes:
     - if_node
     - then_node
    """

    def __init__(self, if_node, then_node):
        super(IfThenNode, self).__init__(nodes={if_node, then_node},
                                         if_node=if_node,
                                         then_node=then_node,
                                         entry_block=if_node.entry_block)

        self.predecessors |= if_node.predecessors
        self.update_predecessors(if_node)

        self.successors |= then_node.successors
        self.update_successors(if_node, exclude={then_node})
        self.update_successors(then_node)


class IfThenElseNode(ControlNode):
    """Represents an if-then-else control structure.

    Distinct nodes A, B, C form an if-then-else control structure if

      (i) A has exactly two successors B and C
      (ii) B and C are single entry, and the union of B and C's successors
           is empty or consists of a single distinct node D

    Note that D is not part of the control structure.

    Attributes:
     - if_node
     - then_node
     - else_node
    """

    def __init__(self, if_node, then_node, else_node):
        super(IfThenElseNode, self).__init__(nodes={if_node, then_node,
                                                    else_node},
                                             if_node=if_node,
                                             then_node=then_node,
                                             else_node=else_node,
                                             entry_block=if_node.entry_block)

        self.predecessors |= if_node.predecessors
        self.update_predecessors(if_node)

        self.successors |= then_node.successors | else_node.successors
        self.update_successors(then_node)
        self.update_successors(else_node)


class UnstructuredIntervalNode(ControlNode):
    """Represents a set of nodes with a single entry point for which no more
    specific high level control structure can be assigned.
    """

    def __init__(self, nodes):
        super(UnstructuredIntervalNode, self).__init__(nodes=set(nodes))
