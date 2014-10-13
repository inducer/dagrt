"""Graph representations"""

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

from leap.vm.language import If


class SimpleIntGraph(object):
    """Maps a graph-like structure to an adjacency list representation
    of a graph with vertices represented by integers."""

    def __init__(self, vertices, edge_fn):
        """
        :arg vertices: The set of vertices
        :arg edge_fn: A function that returns an adjacency list for each vertex
        """
        # Assign a number to each vertex.
        self._vertex_to_num = {}
        self._num_to_vertex = {}
        num_vertices = 0
        for vertex in vertices:
            if vertex not in self._vertex_to_num:
                self._vertex_to_num[vertex] = num_vertices
                self._num_to_vertex[num_vertices] = vertex
                num_vertices += 1
        self._num_vertices = num_vertices

        self._edges = {}
        # Collect edge information on each vertex.
        for vertex in vertices:
            num = self._vertex_to_num[vertex]
            self._edges[num] = frozenset(map(lambda v: self._vertex_to_num[v],
                                             edge_fn(vertex)))

    def __iter__(self):
        return iter(range(0, self._num_vertices))

    def __getitem__(self, num):
        if num not in self._edges:
            raise ValueError('Vertex not found!')
        return self._edges[num]

    def __len__(self):
        return self._num_vertices

    def get_vertex_for_number(self, num):
        if num not in self._num_to_vertex:
            raise ValueError('Vertex not found!')
        return self._num_to_vertex[num]

    def get_number_for_vertex(self, vertex):
        if vertex not in self._vertex_to_num:
            raise ValueError('Vertex not found!')
        return self._vertex_to_num[vertex]


class InstructionDAGIntGraph(SimpleIntGraph):
    """Specialization of SimpleIntGraph that works with instruction DAGs
    (sets of Instructions). Records all the dependency edges in the
    DAG, including conditional dependencies within If statements.
    """

    def __init__(self, dag):
        self._id_to_inst = dict((inst.id, inst) for inst in dag)
        self._ids = self._id_to_inst.keys()

        def edge_func(vertex):
            inst = self._id_to_inst[vertex]
            deps = set(inst.depends_on)
            if isinstance(inst, If):
                deps |= set(inst.then_depends_on)
                deps |= set(inst.else_depends_on)
            return deps

        super(InstructionDAGIntGraph, self).__init__(self._ids, edge_func)

    def get_unconditional_edges(self, vertex):
        """Return the set of vertices that are adjacent to this vertex by an
        unconditional dependency.
        """
        inst = self._id_to_inst[self.get_id_for_number(vertex)]
        return frozenset(map(self.get_number_for_id, inst.depends_on))

    def get_conditional_edges(self, vertex):
        """Return the set of vertices that are adjacent to this vertex by a
        conditional dependency (i.e., a branch of an If statement).
        """
        inst = self._id_to_inst[self.get_id_for_number(vertex)]
        deps = []
        if isinstance(inst, If):
            deps += inst.then_depends_on
            deps += inst.else_depends_on
        return frozenset(map(self.get_number_for_id, deps))

    def get_number_for_vertex(self, vertex):
        return self.get_number_for_id(vertex.id)

    def get_vertex_for_number(self, num):
        return self._id_to_inst[self.get_id_for_number(num)]

    def get_id_for_number(self, num):
        return super(InstructionDAGIntGraph, self).get_vertex_for_number(num)

    def get_number_for_id(self, i):
        return super(InstructionDAGIntGraph, self).get_number_for_vertex(i)
