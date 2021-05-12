__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2014 Matt Wala
"""

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

from pymbolic.mapper.evaluator import EvaluationMapper as EvaluationMapperBase
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.mapper import CombineMapper, IdentityMapper
from pymbolic.mapper.unifier import UnidirectionalUnifier
from pymbolic.primitives import Variable, is_constant
from pymbolic.parser import Parser, _less, _greater, _identifier
from pymbolic.primitives import If as IfThenElse # noqa
from pymbolic.mapper.stringifier import PREC_LOGICAL_OR

import logging
import operator
import pytools.lex
from functools import reduce

logger = logging.getLogger(__name__)


# Precedence constant for IfThenElse.
PREC_IFTHENELSE = PREC_LOGICAL_OR - 1


class ExtendedDependencyMapper(DependencyMapper):
    """Extends DependencyMapper to handle values encountered in dagrt
    IR.
    """

    def map_foreign(self, expr):
        if expr is None or isinstance(expr, str):
            return frozenset()
        else:
            return super().map_foreign(expr)


class EvaluationMapper(EvaluationMapperBase):

    def __init__(self, context, functions):
        """
        :arg context: a mapping from variable names to values
        :arg functions: a mapping from function names to functions
        """
        EvaluationMapperBase.__init__(self, context)
        self.functions = functions

    def map_variable(self, expr):
        if expr.name in self.context:
            return self.context[expr.name]
        elif expr.name in self.functions:
            return self.functions[expr.name]

    def map_generic_call(self, function_name, parameters, kw_parameters):
        if function_name in self.functions:
            function = self.functions[function_name]
        else:
            raise ValueError("Call to unknown function: " + str(function_name))
        evaluated_parameters = tuple(self.rec(param) for param in parameters)
        evaluated_kw_parameters = {
                param_id: self.rec(param)
                for param_id, param in kw_parameters.items()}
        return function(*evaluated_parameters, **evaluated_kw_parameters)

    def map_call(self, expr):
        return self.map_generic_call(expr.function.name, expr.parameters, {})

    def map_call_with_kwargs(self, expr):
        return self.map_generic_call(expr.function.name, expr.parameters,
                                     expr.kw_parameters)


class _ConstantFindingMapper(CombineMapper):
    """Classify subexpressions according to whether they are "constant"
    (have no free variables) or not.
    TODO: CSE caching
    """

    def __init__(self, free_variables):
        self.free_variables = free_variables
        self.node_stack = []

    def __call__(self, expr):
        self.is_constant = {}
        for variable in self.free_variables:
            self.is_constant[variable] = False
        self.node_stack.append(expr)
        CombineMapper.__call__(self, expr)
        return self.is_constant

    def rec(self, expr):
        self.node_stack.append(expr)
        return CombineMapper.rec(self, expr)

    def combine(self, exprs):
        current_expr = self.node_stack.pop()
        result = reduce(operator.and_, exprs)
        self.is_constant[current_expr] = result
        return result

    def map_constant(self, expr):
        self.node_stack.pop()
        self.is_constant[expr] = True
        return True

    map_function_symbol = map_constant

    def map_variable(self, expr):
        self.node_stack.pop()
        result = expr not in self.free_variables
        self.is_constant[expr] = result
        return result


def _is_atomic(expr):
    return isinstance(expr, Variable) or is_constant(expr)


class _ExpressionCollapsingMapper(IdentityMapper):
    """Create a new expression that collapses constant expressions
    (subexpressions with no free variables). Return the new expression
    and an assignment that converts the input to the new expression.
    TODO: CSE caching
    """

    def __init__(self, free_variables):
        self.constant_finding_mapper = _ConstantFindingMapper(free_variables)

    def __call__(self, expr, new_var_func):
        self.new_var_func = new_var_func
        self.is_constant = self.constant_finding_mapper(expr)
        self.assignments = {}
        result = IdentityMapper.__call__(self, expr)
        return result, self.assignments

    def rec(self, expr):
        if _is_atomic(expr) or not self.is_constant[expr]:
            return IdentityMapper.rec(self, expr)
        else:
            new_var = self.new_var_func()
            self.assignments[new_var] = expr
            return new_var

    def map_commut_assoc(self, expr, combine_func):
        # Classify children according to whether they are constant or
        # non-constant. If children are non-constant, it's possible that
        # subexpressions of the children are still constant, so recurse
        # on the non-constant children.
        constants = []
        non_constants = []
        for child in expr.children:
            if self.is_constant[child]:
                constants.append(child)
            else:
                non_constants.append(self.rec(child))

        constants = tuple(constants)
        non_constants = tuple(non_constants)

        # Return the combined sum/product of the constants and
        # non-constants. Take special care to ensure that the
        # constructed sum/product is a binary expression. If not then in
        # place of returning the binary expression return whichever leaf
        # is non-empty.
        if not constants:
            assert non_constants
            if len(non_constants) > 1:
                return combine_func(non_constants)
            else:
                return non_constants[0]

        if len(constants) == 1 and _is_atomic(constants[0]):
            folded_constant = constants[0]
        else:
            new_var = self.new_var_func()
            self.assignments[new_var] = constants[0] \
                if len(constants) == 1 else combine_func(constants)
            folded_constant = new_var

        if non_constants:
            return combine_func(tuple([folded_constant]) + non_constants)
        else:
            return folded_constant

    def map_product(self, expr):
        from pymbolic.primitives import Product
        return self.map_commut_assoc(expr, Product)

    def map_sum(self, expr):
        from pymbolic.primitives import Sum
        return self.map_commut_assoc(expr, Sum)


def collapse_constants(expression, free_variables, assign_func, new_var_func):
    """
    Emit a sequence of calls that assign the constant subexpressions in
    the input to variables.  Return the expression that results from
    collapsing all the constant subexpressions into variables.

    :arg expression: A pymbolic expression
    :arg free_variables: The list of free variables in the expression
    :arg assign_func: A function to call to assign a variable to a constant
                      subexpression
    :arg new_var_func: A function to call to make a new variable
    """
    mapper = _ExpressionCollapsingMapper(free_variables)
    new_expression, variable_map = mapper(expression, new_var_func)
    for variable, expr in variable_map.items():
        assign_func(variable, expr)
    return new_expression


class _ExtendedUnifier(UnidirectionalUnifier):
    """
    This class extends the unification mapper as follows:
       - Handles terms with function calls with keyword arguments
       - Supports unifying function symbols
       - Limited support for unification modulo identity in
         binary addition or multiplication
         (i.e. allows (x*c, c) to be unified with x=1)
    """

    def map_call(self, expr, other, urecs):
        if not isinstance(expr, type(other)):
            return []

        expr_parameters = expr.parameters
        other_parameters = other.parameters

        # Unify parameters
        if len(expr_parameters) != len(other_parameters):
            return []

        from pymbolic.primitives import CallWithKwargs

        if isinstance(expr, CallWithKwargs):
            from operator import itemgetter

            if set(expr.kw_parameters.keys()) != set(other.kw_parameters.keys()):
                return []

            expr_parameters += tuple(val for key, val in
                                     sorted(expr.kw_parameters.items(),
                                            key=itemgetter(0)))
            other_parameters += tuple(val for key, val in
                                      sorted(other.kw_parameters.items(),
                                             key=itemgetter(0)))

        for expr_param, other_param in zip(expr_parameters, other_parameters):
            urecs = self.rec(expr_param, other_param, urecs)

        # Unify function symbols
        urecs = self.rec(expr.function, other.function, urecs)

        return urecs

    map_call_with_kwargs = map_call

    def map_modulo_identity(self, expr, other, urecs, mapper, id_element):
        """
        :arg mapper: mapper to call once done
        :arg id_element: identity element to add

        Currently this is restricted to the case that "expr" is a binary
        expression and "other" is a single value.
        """
        # Only apply this when other is a single element and expr is multiple
        # elements.
        if len(expr.children) != 2 or hasattr(other, "children"):
            return mapper(expr, other, urecs)

        variables = {
            term for term in expr.children
            if isinstance(term, Variable)
            and term.name in self.lhs_mapping_candidates}

        from pymbolic.mapper.unifier import unify_many

        # Try matching each free variable in the expression with the identity
        # element.
        new_urecs = []
        new_other = type(expr)((id_element, other))
        for variable in variables:
            urec = self.unification_record_from_equation(variable, id_element)
            new_urecs.extend(mapper(expr, new_other, unify_many(urecs, urec)))

        return new_urecs

    def map_sum(self, expr, other, urecs):
        mapper = super().map_sum
        return self.map_modulo_identity(expr, other, urecs, mapper, 0)

    def map_product(self, expr, other, urecs):
        mapper = super().map_product
        return self.map_modulo_identity(expr, other, urecs, mapper, 1)


def match(template, expression, free_variable_names=None,
          bound_variable_names=None, pre_match=None):
    """Attempt to match the free variables found in `template` to terms in
    `expression`, modulo associativity and commutativity.

    This implements a one-way unification algorithm, matching free
    variables in `template` to subexpressions of `expression`.

    If `free_variable_names` is *None*, then all variables except those in
    `bound_variable_names` are treated as free.

    Matches that are already known to hold can be specified in `pre_match`, a
    map from variable names to subexpressions (or strings representing
    subexpressions).

    Return a map from variable names in `free_variable_names` to
    expressions.
    """
    if isinstance(template, str):
        template = parse(template)

    if isinstance(expression, str):
        expression = parse(expression)

    if bound_variable_names is None:
        bound_variable_names = set()

    if free_variable_names is None:
        from dagrt.utils import get_variables
        free_variable_names = get_variables(
            template, include_function_symbols=True)
        free_variable_names -= set(bound_variable_names)

    urecs = None
    if pre_match is not None:
        eqns = []
        for name, expr in pre_match.items():
            if name not in free_variable_names:
                raise ValueError(
                    "'%s' was given in 'pre_match' but is "
                    "not a candidate for matching" % name)
            if isinstance(expr, str):
                expr = parse(expr)
            eqns.append((Variable(name), expr))
        from pymbolic.mapper.unifier import UnificationRecord
        urecs = [UnificationRecord(eqns)]

    unifier = _ExtendedUnifier(free_variable_names)
    records = unifier(template, expression, urecs)

    if len(records) > 1:
        from warnings import warn
        warn('Matching\n"{expr}"\nto\n"{template}"\n'
             "is ambiguous - using first match".format(
                 expr=expression, template=template))

    if not records:
        raise ValueError("Cannot unify expressions.")

    return {key.name: val for key, val in records[0].equations}


def _hack_lex_table(lex_table):
    new_lex_table = []
    for entry in lex_table:
        if entry[0] == "identifier":
            # Allow backticks to delimit identifiers.
            entry = ("identifier", ("|", entry[1],
                    pytools.lex.RE("`[<>:a-zA-Z0-9_]*`")))
        new_lex_table.append(entry)
    return new_lex_table


class _ExtendedParser(Parser):
    def parse_terminal(self, pstate):
        import pymbolic.primitives as primitives

        next_tag = pstate.next_tag()
        if next_tag is _less:
            # tagged identifier: <func>something
            identifier = pstate.next_str_and_advance()
            pstate.expect(_identifier)
            identifier += pstate.next_str_and_advance()
            pstate.expect(_greater)
            identifier += pstate.next_str_and_advance()
            if pstate.is_next(_identifier):
                identifier += pstate.next_str_and_advance()

            return primitives.Variable(identifier)
        else:
            return super().parse_terminal(pstate)

    lex_table = _hack_lex_table(Parser.lex_table)


def parse(expr):
    """Return a pymbolic expression constructed from the string.

    Values between backticks ("`") are parsed as variable names.
    Tagged identifiers ("<func>f") are also parsed as variable names.
    """
    from pymbolic import var

    def remove_backticks(expr):
        if not isinstance(expr, var):
            return expr
        varname = expr.name
        if varname.startswith("`") and varname.endswith("`"):
            return var(varname[1:-1])
        return expr

    from pymbolic.mapper.substitutor import SubstitutionMapper
    parser = _ExtendedParser()
    substitutor = SubstitutionMapper(remove_backticks)
    return substitutor(parser(expr))


def substitute(expression, variable_assignments=None, **kwargs):
    """Perform variable substitution.

    :arg expression: A string or :mod:`pymbolic` expression.
        If a string, it will be parsed with :func:`parse`.
    :arg variable_assignments: Mapping from variable names to expressions
    :arg kwargs: Extra arguments passed to to :func:`pymbolic.substitute`
    """
    if variable_assignments is None:
        variable_assignments = {}

    from pymbolic import substitute as substitute_pymbolic

    if isinstance(expression, str):
        expression = parse(expression)

    return substitute_pymbolic(expression, variable_assignments, **kwargs)


# vim: foldmethod=marker
