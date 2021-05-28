"""Some generic DAG transformation passes"""


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

from pymbolic.mapper import IdentityMapper
from pytools import UniqueNameGenerator
from dagrt.codegen.dag_ast import (
        ASTIdentityMapper, get_statements_in_ast,
        Block, StatementWrapper)

__doc__ = """
.. autofunction:: eliminate_self_dependencies
.. autofunction:: isolate_function_arguments
.. autofunction:: isolate_function_calls
.. autofunction:: expand_IfThenElse
"""


def get_stmt_id_generator(statements):
    return UniqueNameGenerator({stmt.id for stmt in statements})


def get_var_name_generator(statements):
    existing_variables = set()
    for stmt in statements:
        existing_variables.update(stmt.get_written_variables())
        existing_variables.update(stmt.get_read_variables())
    return UniqueNameGenerator(existing_variables)


# {{{ ast statement rewriter

class ASTStatementRewriter(ASTIdentityMapper):
    def __init__(self, stmt_id_gen, var_name_gen):
        self.stmt_id_gen = stmt_id_gen
        self.var_name_gen = var_name_gen

    def map_StatementWrapper(self, expr):
        new_statements = [
                StatementWrapper(stmt)
                for stmt in self.map_statement(expr.statement)]

        if len(new_statements) > 1:
            return Block(*new_statements)
        else:
            return new_statements[0]

    def map_statement(self, statement):
        raise NotImplementedError()


def apply_statement_rewriter(rewriter_cls, phase_ast):
    statements = list(get_statements_in_ast(phase_ast))
    rewriter = rewriter_cls(
            stmt_id_gen=get_stmt_id_generator(statements),
            var_name_gen=get_var_name_generator(statements))

    return rewriter(phase_ast)

# }}}


# {{{ eliminate self dependencies

class SelfDependencyEliminator(ASTStatementRewriter):
    def map_statement(self, stmt):
        read_and_written = (
                stmt.get_read_variables() & stmt.get_written_variables())

        if not read_and_written:
            return [stmt]

        substs = []
        tmp_stmt_ids = []

        new_statements = []
        from dagrt.language import Assign
        from pymbolic import var
        for var_name in read_and_written:
            tmp_var_name = self.var_name_gen(
                    "temp_"
                    + var_name.replace("<", "_").replace(">", "_"))
            substs.append((var_name, var(tmp_var_name)))

            tmp_stmt_id = self.stmt_id_gen("temp")
            tmp_stmt_ids.append(tmp_stmt_id)

            new_tmp_stmt = Assign(
                    tmp_var_name, (), var(var_name),
                    condition=stmt.condition,
                    id=tmp_stmt_id,
                    depends_on=stmt.depends_on)
            new_statements.append(new_tmp_stmt)

        from pymbolic import substitute
        new_stmt = (stmt
                .map_expressions(
                    lambda expr: substitute(expr, dict(substs)),
                    include_lhs=False)
                .copy(
                    # lhs will be rewritten, but we don't want that.
                    depends_on=stmt.depends_on | frozenset(tmp_stmt_ids)))
        new_statements.append(new_stmt)

        return new_statements


def eliminate_self_dependencies(phase_ast):
    return apply_statement_rewriter(SelfDependencyEliminator, phase_ast)

# }}}


# {{{ isolate function arguments

class ExprFunctionArgumentIsolator(IdentityMapper):
    def __init__(self, new_statements,
            stmt_id_gen, var_name_gen):
        super().__init__()
        self.new_statements = new_statements
        self.stmt_id_gen = stmt_id_gen
        self.var_name_gen = var_name_gen

    def isolate_arg(self, expr, base_condition, base_deps, extra_deps):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return expr

        # FIXME: These aren't awesome identifiers.
        tmp_var_name = self.var_name_gen("tmp")

        tmp_stmt_id = self.stmt_id_gen("tmp")
        extra_deps.append(tmp_stmt_id)

        sub_extra_deps = []
        rec_result = self.rec(
                expr, base_condition, base_deps, sub_extra_deps)

        from dagrt.language import Assign
        new_stmt = Assign(
                tmp_var_name, (), rec_result,
                condition=base_condition,
                depends_on=base_deps | frozenset(sub_extra_deps),
                id=tmp_stmt_id)

        self.new_statements.append(new_stmt)

        from pymbolic import var
        return var(tmp_var_name)

    def map_call(self, expr, base_condition, base_deps, extra_deps):
        return type(expr)(
                expr.function,
                tuple(self.isolate_arg(child, base_condition, base_deps, extra_deps)
                    for child in expr.parameters))

    def map_call_with_kwargs(self, expr, base_condition, base_deps, extra_deps):
        return type(expr)(
                expr.function,
                tuple(self.isolate_arg(child, base_condition, base_deps, extra_deps)
                    for child in expr.parameters),
                {key: self.isolate_arg(val, base_condition, base_deps, extra_deps)
                    for key, val in sorted(expr.kw_parameters.items())}
                )


class StatementFunctionArgumentIsolator(ASTStatementRewriter):
    def map_statement(self, stmt):
        new_statements = []

        fai = ExprFunctionArgumentIsolator(
                new_statements=new_statements,
                stmt_id_gen=self.stmt_id_gen,
                var_name_gen=self.var_name_gen)

        base_deps = stmt.depends_on
        new_deps = []

        new_statements.append(
                stmt
                .map_expressions(
                    lambda expr: fai(
                        expr, stmt.condition, base_deps, new_deps))
                .copy(depends_on=stmt.depends_on | frozenset(new_deps)))

        return new_statements


def isolate_function_arguments(phase_ast):
    return apply_statement_rewriter(StatementFunctionArgumentIsolator, phase_ast)

# }}}


# {{{ isolate function calls

class ExpressionFunctionCallIsolator(IdentityMapper):
    def __init__(self, new_statements,
            stmt_id_gen, var_name_gen):
        super().__init__()
        self.new_statements = new_statements
        self.stmt_id_gen = stmt_id_gen
        self.var_name_gen = var_name_gen

    def isolate_call(self, expr, base_condition, base_deps, extra_deps,
                     super_method):
        # FIXME: These aren't awesome identifiers.
        tmp_var_name = self.var_name_gen("tmp")

        tmp_stmt_id = self.stmt_id_gen("tmp")
        extra_deps.append(tmp_stmt_id)

        sub_extra_deps = []
        rec_result = super_method(
                expr, base_deps, sub_extra_deps)

        from pymbolic.primitives import Call, CallWithKwargs
        assert isinstance(rec_result, (Call, CallWithKwargs))

        parameters = []
        kw_parameters = {}

        for par in rec_result.parameters:
            parameters.append(par)

        if isinstance(rec_result, CallWithKwargs):
            for par_name, par in rec_result.kw_parameters.items():
                kw_parameters[par_name] = par

        from dagrt.language import AssignFunctionCall
        new_stmt = AssignFunctionCall(
                assignees=(tmp_var_name,),
                function_id=rec_result.function.name,
                parameters=tuple(parameters),
                kw_parameters=kw_parameters,
                id=tmp_stmt_id,
                condition=base_condition,
                depends_on=base_deps | frozenset(sub_extra_deps))

        self.new_statements.append(new_stmt)

        from pymbolic import var
        return var(tmp_var_name)

    def map_call(self, expr, base_condition, base_deps, extra_deps):
        return self.isolate_call(
                expr, base_condition, base_deps, extra_deps,
                super().map_call)

    def map_call_with_kwargs(self, expr, base_condition, base_deps, extra_deps):
        return self.isolate_call(
                expr, base_condition, base_deps, extra_deps,
                super()
                .map_call_with_kwargs)


class StatementFunctionCallIsolator(ASTStatementRewriter):
    def map_statement(self, stmt):
        from dagrt.language import Assign
        if not isinstance(stmt, Assign):
            return [stmt]

        new_deps = []
        new_statements = []

        fci = ExpressionFunctionCallIsolator(
                new_statements=new_statements,
                stmt_id_gen=self.stmt_id_gen,
                var_name_gen=self.var_name_gen)

        new_statements.append(
                stmt
                .map_expressions(
                    lambda expr: fci(
                        expr, stmt.condition, stmt.depends_on, new_deps))
                .copy(depends_on=stmt.depends_on | frozenset(new_deps)))
        from pymbolic.primitives import Call, CallWithKwargs
        assert not isinstance(new_statements[-1].rhs,
                (Call, CallWithKwargs))

        return new_statements


def isolate_function_calls(phase_ast):
    return apply_statement_rewriter(StatementFunctionCallIsolator, phase_ast)

# }}}


def flat_LogicalAnd(*children):  # noqa
    from pymbolic.primitives import LogicalAnd
    result = []
    for child in children:
        if isinstance(child, LogicalAnd):
            result.extend(child.children)
        else:
            result.append(child)
    return LogicalAnd(tuple(result))


# {{{ expand IfThenElse expressions

class ExprIfThenElseExpander(IdentityMapper):

    def __init__(self, new_statements, stmt_id_gen, var_name_gen):
        super().__init__()
        self.new_statements = new_statements
        self.stmt_id_gen = stmt_id_gen
        self.var_name_gen = var_name_gen

    def map_if(self, expr, base_condition, base_deps, extra_deps):
        from pymbolic.primitives import LogicalNot
        from pymbolic import var

        flag = var(self.var_name_gen("<cond>ifthenelse_cond"))
        tmp_result = self.var_name_gen("ifthenelse_result")
        if_stmt_id = self.stmt_id_gen("ifthenelse_cond")
        then_stmt_id = self.stmt_id_gen("ifthenelse_then")
        else_stmt_id = self.stmt_id_gen("ifthenelse_else")

        sub_condition_deps = []
        rec_condition = self.rec(expr.condition, base_condition, base_deps,
                                 sub_condition_deps)

        sub_then_deps = []
        then_condition = flat_LogicalAnd(base_condition, flag)
        rec_then = self.rec(expr.then, then_condition,
                            base_deps | frozenset([if_stmt_id]), sub_then_deps)

        sub_else_deps = []
        else_condition = flat_LogicalAnd(base_condition, LogicalNot(flag))
        rec_else = self.rec(expr.else_, else_condition,
                            base_deps | frozenset([if_stmt_id]), sub_else_deps)

        from dagrt.language import Assign

        self.new_statements.extend([
            Assign(
                assignee=flag.name,
                assignee_subscript=(),
                expression=rec_condition,
                condition=base_condition,
                id=if_stmt_id,
                depends_on=base_deps | frozenset(sub_condition_deps)),
            Assign(
                assignee=tmp_result,
                assignee_subscript=(),
                condition=then_condition,
                expression=rec_then,
                id=then_stmt_id,
                depends_on=(
                    base_deps
                    | frozenset(sub_then_deps)
                    | frozenset([if_stmt_id]))),
            Assign(
                assignee=tmp_result,
                assignee_subscript=(),
                condition=else_condition,
                expression=rec_else,
                id=else_stmt_id,
                depends_on=(
                    base_deps
                    | frozenset(sub_else_deps)
                    | frozenset([if_stmt_id])))
                ])

        extra_deps.extend([then_stmt_id, else_stmt_id])
        return var(tmp_result)


class StatementIfThenElseExpander(ASTStatementRewriter):
    def map_statement(self, stmt):
        new_statements = []

        expander = ExprIfThenElseExpander(
            new_statements=new_statements,
            stmt_id_gen=self.stmt_id_gen,
            var_name_gen=self.var_name_gen)

        base_deps = stmt.depends_on
        new_deps = []

        new_statements.append(
            stmt.map_expressions(
                lambda expr: expander(expr, stmt.condition, base_deps, new_deps))
            .copy(depends_on=stmt.depends_on | frozenset(new_deps)))

        return new_statements


def expand_IfThenElse(phase_ast):  # noqa
    """
    Turn IfThenElse expressions into values that are computed as a result of an
    If statement. This is useful for targets that do not support ternary
    operators.
    """
    return apply_statement_rewriter(StatementIfThenElseExpander, phase_ast)

# }}}

# vim: foldmethod=marker
