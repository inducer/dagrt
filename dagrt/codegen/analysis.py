"""Analysis passes"""

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

from pymbolic.mapper import Collector
from dagrt.language import YieldState, SwitchPhase, AssignFunctionCall


# {{{ verifier


def _quote(string):
    return f'"{string}"'


def verify_switch_phases(phases, errors):
    """
    Ensure that phases referenced by :class:`SwitchPhase` exist.

    :arg statements: A set of statements to verify
    :arg phases: A map from phase names to phases
    :arg errors: An error list to which new errors get appended
    """
    for phase in phases.values():
        for inst in phase.statements:
            if not isinstance(inst, SwitchPhase):
                continue
            if inst.next_phase not in phases:
                errors.append(
                    'Phase "{}" referenced by statement "{}:{}" not found'
                    .format(inst.next_phase, phase, inst))


def verify_all_dependencies_exist(phases, errors):
    """
    Ensure that all statement dependencies exist.

    :arg statements: A set of statements to verify
    :arg phases: A map from phase names to phases
    :arg errors: An error list to which new errors get appended
    """
    ids = {inst.id
            for phase in phases.values()
            for inst in phase.statements}

    # Check statements
    for phase in phases.values():
        for inst in phase.statements:
            deps = set(inst.depends_on)
            if not deps <= ids:
                errors.extend(
                    ['Dependency "{}" referenced by statement "{}" not found'
                     .format(dep_name, inst) for dep_name in deps - ids])

    # Check phases.
    for phase_name, phase in phases.items():
        deps = set(phase.depends_on)
        if not deps <= ids:
            errors.extend(
                ['Dependencies {} referenced by phase "{}" not found'
                 .format(", ".join(_quote(dep) for dep in ids - deps),
                         phase_name)])


def verify_no_circular_dependencies(statements, errors):
    """
    Ensure that there are no circular dependencies among the statements.

    :arg statements: A set of statements to verify
    :arg errors: An error list to which new errors get appended
    """
    id_to_statement = {inst.id: inst for inst in statements}
    stack = list(statements)
    visiting = set()
    visited = set()
    while stack:
        top = stack[-1]
        if top.id not in visited:
            visited.add(top.id)
            visiting.add(top.id)
            for neighbor in top.depends_on:
                if neighbor in visiting:
                    errors.append("Circular dependency chain found")
                    return
                stack.append(id_to_statement[neighbor])
        else:
            if top.id in visiting:
                visiting.remove(top.id)
            stack.pop()


def verify_single_definition_cond_rule(statements, errors):
    """
    Verify that <cond> variables are never redefined.

    :arg statements: A set of statements to verify
    :arg errors: An error list to which new errors get appended
    """
    cond_variables = {}

    for statement in statements:
        for varname in statement.get_written_variables():
            if not varname.startswith("<cond>"):
                continue
            if varname not in cond_variables:
                cond_variables[varname] = [statement]
            else:
                cond_variables[varname].append(statement)

    for varname, insts in cond_variables.items():
        if len(insts) > 1:
            errors.append(
                'Conditional variable "{}" defined by multiple statements: {}'
                .format(varname, ", ".join(_quote(str(inst)) for inst in insts)))


class CodeGenerationError(Exception):
    def __init__(self, errors):
        self.errors = errors

    def __str__(self):
        return "Errors encountered in input to code generator.\n" + \
            "\n".join(self.errors)


def verify_code(code):
    """Verify that the DAG is well-formed."""
    errors = []

    try:
        # Wrap in a try block, since some verifier passes may fail due to badly
        # malformed code.
        verify_all_dependencies_exist(code.phases, errors)
        for phase in code.phases.values():
            verify_no_circular_dependencies(phase.statements, errors)

        verify_switch_phases(code.phases, errors)

        for phase in code.phases.values():
            verify_single_definition_cond_rule(phase.statements, errors)

    except Exception as e:
        # Ensure there is at least one error to report.
        if len(errors) == 0:
            raise e

    if errors:
        raise CodeGenerationError(errors)

# }}}


# {{{ collect function names from DAG

class _FunctionNameCollector(Collector):

    def map_variable(self, expr):
        if expr.name.startswith("<func>"):
            return {expr.name}
        return set()

    def map_call(self, expr):
        return ({expr.function.name}
                | super().map_call(expr))

    def map_call_with_kwargs(self, expr):
        return ({expr.function.name}
                | super().map_call_with_kwargs(expr))


def collect_function_names_from_dag(dag, no_expressions=False):
    """
    :arg no_expressions: Do not consider expressions when finding function names.
        This saves a bit of time if, for example,
        :func:`dagrt.codegen.transform.isolate_function_calls` has been called
        on *dag*.
    """
    fnc = _FunctionNameCollector()

    result = set()

    def mapper(expr):
        result.update(fnc(expr))
        return expr
    for phase in dag.phases.values():
        for stmt in phase.statements:
            if isinstance(stmt, AssignFunctionCall):
                result.add(stmt.function_id)

            if not no_expressions:
                stmt.map_expressions(mapper)

    return result

# }}}


# {{{ collect time IDs from DAG

def collect_time_ids_from_dag(dag):
    result = set()

    for phase in dag.phases.values():
        for stmt in phase.statements:
            if isinstance(stmt, YieldState):
                result.add(stmt.time_id)

    return result

# }}}


# {{{ collect ODE component names from DAG

def collect_ode_component_names_from_dag(dag):
    result = set()

    for phase in dag.phases.values():
        for stmt in phase.statements:
            if isinstance(stmt, YieldState):
                result.add(stmt.component_id)

    return result

# }}}


# {{{ variable to last dependent statement mapping

def var_to_last_dependent_statement_mapping(names, statement_lists):
    """For each function in names, return a mapping of each variable to the
    latest statement in statement_lists at which that variable is used.
    This is used for intermediate deallocation of variables that no longer
    need to be read or written.

    :arg names: a list of function names in the ast.
    :arg statement_lists: a set of topological orderings of the statements
        in each function.
    """

    tbl = {}

    for name, stmts in zip(names, statement_lists):

        for statement in stmts:
            # Associate latest statement in this phase at which
            # a given variable is used
            read_and_written = statement.get_read_variables().union(
                    statement.get_written_variables())
            for variable in read_and_written:
                tbl[variable, name] = statement.id

    return tbl

# }}}

# vim: foldmethod=marker
