"""Base class for code generators"""

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

from dagrt.codegen.dag_ast import \
        Block, IfThen, IfThenElse, StatementWrapper, ForLoop


class StructuredCodeGenerator:
    """Code generation for structured languages"""

    def __call__(self, code):
        """Return *code* as a generated code string."""
        raise NotImplementedError()

    def lower_ast(self, ast):
        self.lower_node(ast)
        self.emit_return()

    def lower_node(self, node):
        """
        :arg node: An AST node to lower
        """
        if isinstance(node, StatementWrapper):
            self.lower_inst(node.statement)

        elif isinstance(node, IfThen):
            self.emit_if_begin(node.condition)
            self.lower_node(node.then)
            self.emit_if_end()

        elif isinstance(node, IfThenElse):
            self.emit_if_begin(node.condition)
            self.lower_node(node.then)
            self.emit_else_begin()
            self.lower_node(node.else_)
            self.emit_if_end()

        elif isinstance(node, ForLoop):
            self.emit_for_begin(node.loop_var_name, node.lbound, node.ubound)
            self.lower_node(node.body)
            self.emit_for_end(node.loop_var_name)

        elif isinstance(node, Block):
            for child in node.children:
                self.lower_node(child)

        else:
            raise ValueError("Unrecognized node type {type}".format(
                type=type(node).__name__))

    def lower_inst(self, inst):
        """Emit the code for an statement."""

        method_name = "emit_inst_"+type(inst).__name__
        try:
            method = getattr(self, method_name)
        except AttributeError:
            raise RuntimeError(
                    "{gen} cannot handle statement of type {inst}"
                    .format(
                        gen=repr(type(self)),
                        inst=type(inst).__name__))

        return method(inst)

    # Emit routines (to be implemented by subclass, in addition to emit_inst_)

    def begin_emit(self, dag):
        raise NotImplementedError()

    def finish_emit(self, dag):
        raise NotImplementedError()

    def emit_def_begin(self, name):
        raise NotImplementedError()

    def emit_def_end(self):
        raise NotImplementedError()

    def emit_if_begin(self, expr):
        raise NotImplementedError()

    def emit_if_end(self):
        raise NotImplementedError()

    def emit_else_begin(self):
        raise NotImplementedError()

    def emit_for_begin(self, loop_var_name, lbound, ubount):
        raise NotImplementedError()

    def emit_for_end(self, loop_var_name):
        raise NotImplementedError()

    def emit_return(self):
        raise NotImplementedError()
