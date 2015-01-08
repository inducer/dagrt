"""Some generic DAG transformation passes"""

from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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

import six
from pymbolic.mapper import IdentityMapper


# {{{ eliminate self dependencies

def eliminate_self_dependencies(dag):
    insn_id_gen = dag.get_insn_id_generator()
    var_name_gen = dag.get_var_name_generator()

    new_instructions = []
    for insn in dag.instructions:
        read_and_written = insn.get_read_variables() & insn.get_assignees()

        if not read_and_written:
            new_instructions.append(insn)
            continue

        substs = []
        tmp_insn_ids = []

        from leap.vm.language import AssignExpression
        from pymbolic import var
        for var_name in read_and_written:
            tmp_var_name = var_name_gen("temp_" + var_name)
            substs.append((var_name, var(tmp_var_name)))

            tmp_insn_id = insn_id_gen("temp")
            tmp_insn_ids.append(tmp_insn_id)

            new_tmp_insn = AssignExpression(
                    tmp_var_name, var(var_name),
                    id=tmp_insn_id,
                    depends_on=insn.depends_on)
            new_instructions.append(new_tmp_insn)

        from pymbolic import substitute
        new_insn = (insn
                .map_expressions(
                    lambda expr: substitute(insn.expression, dict(substs)))
                .copy(depends_on=frozenset(tmp_insn_ids)))

        new_instructions.append(new_insn)

    return dag.copy(instructions=new_instructions)

# }}}


# {{{ isolate function arguments

class FunctionArgumentIsolator(IdentityMapper):
    def __init__(self, new_instructions,
            insn_id_gen, var_name_gen):
        super(IdentityMapper, self).__init__()
        self.new_instructions = new_instructions
        self.insn_id_gen = insn_id_gen
        self.var_name_gen = var_name_gen

    def isolate_arg(self, expr, base_deps, extra_deps):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return expr

        # FIXME: These aren't awesome identifiers.
        tmp_var_name = self.var_name_gen("tmp")

        tmp_insn_id = self.insn_id_gen("tmp")
        extra_deps.append(tmp_insn_id)

        sub_extra_deps = []
        rec_result = self.rec(
                expr, base_deps, sub_extra_deps)

        from leap.vm.language import AssignExpression
        new_insn = AssignExpression(
                tmp_var_name, rec_result,
                depends_on=base_deps | frozenset(sub_extra_deps),
                id=tmp_insn_id)

        self.new_instructions.append(new_insn)

        from pymbolic import var
        return var(tmp_var_name)

    def map_call(self, expr, base_deps, extra_deps):
        return type(expr)(
                expr.function,
                tuple(self.isolate_arg(child, base_deps, extra_deps)
                    for child in expr.parameters))

    def map_call_with_kwargs(self, expr, base_deps, extra_deps):
        return type(expr)(
                expr.function,
                tuple(self.isolate_arg(child, base_deps, extra_deps)
                    for child in expr.parameters),
                dict(
                    (key, self.isolate_arg(val, base_deps, extra_deps))
                    for key, val in six.iteritems(expr.kw_parameters))
                    )


def isolate_function_arguments(dag):
    insn_id_gen = dag.get_insn_id_generator()
    var_name_gen = dag.get_var_name_generator()

    new_instructions = []

    fai = FunctionArgumentIsolator(
            new_instructions=new_instructions,
            insn_id_gen=insn_id_gen,
            var_name_gen=var_name_gen)

    for insn in dag.instructions:
        base_deps = insn.depends_on
        new_deps = []

        new_instructions.append(
                insn
                .map_expressions(
                    lambda expr: fai(
                        expr, base_deps, new_deps))
                .copy(depends_on=insn.depends_on | frozenset(new_deps)))

    return dag.copy(instructions=new_instructions)

# }}}


# {{{ isolate function calls

class FunctionCallIsolator(IdentityMapper):
    def __init__(self, new_instructions,
            insn_id_gen, var_name_gen):
        super(IdentityMapper, self).__init__()
        self.new_instructions = new_instructions
        self.insn_id_gen = insn_id_gen
        self.var_name_gen = var_name_gen

    def isolate_call(self, expr, base_deps, extra_deps, super_method):
        # FIXME: These aren't awesome identifiers.
        tmp_var_name = self.var_name_gen("tmp")

        tmp_insn_id = self.insn_id_gen("tmp")
        extra_deps.append(tmp_insn_id)

        sub_extra_deps = []
        rec_result = super_method(
                expr, base_deps, sub_extra_deps)

        from leap.vm.language import AssignExpression
        new_insn = AssignExpression(
                tmp_var_name, rec_result, id=tmp_insn_id,
                depends_on=base_deps | frozenset(sub_extra_deps))

        self.new_instructions.append(new_insn)

        from pymbolic import var
        return var(tmp_var_name)

    def map_call(self, expr, base_deps, extra_deps):
        return self.isolate_call(
                expr, base_deps, extra_deps,
                super(FunctionCallIsolator, self).map_call)

    def map_call_with_kwargs(self, expr, base_deps, extra_deps):
        return self.isolate_call(
                expr, base_deps, extra_deps,
                super(FunctionCallIsolator, self)
                .map_call_with_kwargs)


def isolate_function_calls(dag):
    """
    :func:`isolate_function_arguments` should be
    called before this.
    """

    insn_id_gen = dag.get_insn_id_generator()
    var_name_gen = dag.get_var_name_generator()

    new_instructions = []

    fci = FunctionCallIsolator(
            new_instructions=new_instructions,
            insn_id_gen=insn_id_gen,
            var_name_gen=var_name_gen)

    from pymbolic.primitives import Call, CallWithKwargs
    for insn in dag.instructions:
        base_deps = insn.depends_on
        new_deps = []

        from leap.vm.language import AssignExpression
        if (isinstance(insn, AssignExpression)
                and isinstance(insn.expression, (Call, CallWithKwargs))):
            new_instructions.append(insn)
        else:
            new_instructions.append(
                    insn
                    .map_expressions(
                        lambda expr: fci(
                            expr, base_deps, new_deps))
                    .copy(depends_on=insn.depends_on | frozenset(new_deps)))

    return dag.copy(instructions=new_instructions)

# }}}

# vim: foldmethod=marker
