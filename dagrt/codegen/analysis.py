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

import six
from pymbolic.mapper import Collector
from dagrt.language import YieldState, PhaseTransition, AssignFunctionCall


# {{{ verifier


def _quote(string):
    return "\"{0}\"".format(string)


def verify_phase_transitions(phases, errors):
    """
    Ensure that phases referenced by PhaseTransition exist.

    :arg statements: A set of statements to verify
    :arg phases: A map from phase names to phases
    :arg errors: An error list to which new errors get appended
    """
    phase_names = [key for key in phases.keys()]
    for phase in six.itervalues(phases):
        for inst in phase.statements:
            if not isinstance(inst, PhaseTransition):
                continue
            if inst.next_phase not in phase_names:
                errors.append(
                    "Phase \"{0}\" referenced by statement \"{1}:{2}\" not found"
                    .format(inst.next_phase, phase, inst))


def verify_all_dependencies_exist(phases, errors):
    """
    Ensure that all statement dependencies exist.

    :arg statements: A set of statements to verify
    :arg phases: A map from phase names to phases
    :arg errors: An error list to which new errors get appended
    """
    ids = set(inst.id
            for phase in six.itervalues(phases)
            for inst in phase.statements)

    # Check statements
    for phase in six.itervalues(phases):
        for inst in phase.statements:
            deps = set(inst.depends_on)
            if not deps <= ids:
                errors.extend(
                    ["Dependency \"{0}\" referenced by statement \"{1}\" not found"
                     .format(dep_name, inst) for dep_name in deps - ids])

    # Check phases.
    for phase_name, phase in six.iteritems(phases):
        deps = set(phase.depends_on)
        if not deps <= ids:
            errors.extend(
                ["Dependencies {0} referenced by phase \"{1}\" not found"
                 .format(", ".join(_quote(dep) for dep in ids - deps),
                         phase_name)])


def verify_no_circular_dependencies(statements, errors):
    """
    Ensure that there are no circular dependencies among the statements.

    :arg statements: A set of statements to verify
    :arg errors: An error list to which new errors get appended
    """
    id_to_statement = dict((inst.id, inst) for inst in statements)
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

    import six
    for varname, insts in six.iteritems(cond_variables):
        if len(insts) > 1:
            errors.append(
                "Conditional variable \"{0}\" defined by multiple statements: {1}"
                .format(varname, ", ".join(_quote(str(inst)) for inst in insts)))


class CodeGenerationError(Exception):
    def __init__(self, errors):
        self.errors = errors

    def __str__(self):
        return 'Errors encountered in input to code generator.\n' + \
            '\n'.join(self.errors)


def verify_code(code):
    """Verify that the DAG is well-formed."""
    errors = []

    try:
        # Wrap in a try block, since some verifier passes may fail due to badly
        # malformed code.
        verify_all_dependencies_exist(code.phases, errors)
        for phase in six.itervalues(code.phases):
            verify_no_circular_dependencies(phase.statements, errors)

        verify_phase_transitions(code.phases, errors)

        for phase in six.itervalues(code.phases):
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
            return set([expr.name])
        return set()

    def map_call(self, expr):
        return (set([expr.function])
                | super(_FunctionNameCollector, self).map_call(expr))

    def map_call_with_kwargs(self, expr):
        return (set([expr.function])
                | super(_FunctionNameCollector, self).map_call_with_kwargs(expr))


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
    for phase in six.itervalues(dag.phases):
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

    for phase in six.itervalues(dag.phases):
        for stmt in phase.statements:
            if isinstance(stmt, YieldState):
                result.add(stmt.time_id)

    return result

# }}}


# {{{ collect ODE component names from DAG

def collect_ode_component_names_from_dag(dag):
    result = set()

    for phase in six.itervalues(dag.phases):
        for stmt in phase.statements:
            if isinstance(stmt, YieldState):
                result.add(stmt.component_id)

    return result

# }}}


# {{{ variable dependency finder

def var_to_statement_table(names, functions):

        """Return a table describing variable dependencies
           in terms of statement ids.
        """

        tbl = {}

        from dagrt.codegen.dag_ast import get_statements_in_ast

        for name, func in zip(names, functions):

            for statement in get_statements_in_ast(func):
                # Associate latest statement in this phase at which
                # a given variable is used
                read_and_written = statement.get_read_variables().union(
                        statement.get_written_variables())
                for variable in read_and_written:
                    tbl[variable, name] = statement.id

        return tbl

# }}}

# vim: foldmethod=marker
