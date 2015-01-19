"""Implicit solver utilities"""

__copyright__ = """
Copyright (C) 2014, 2015 Matt Wala
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

from leap.vm.function_registry import base_function_registry
from pymbolic import var


class ScipySolverGenerator(object):
    """The purpose of this class is to create a default implementation of a solver
    function that can be called at runtime by the Python code generator or
    interpreter to perform an implicit solve, as well as to provide the code that
    describes how to call the solver function.

    `ScipySolverGenerator` creates a solver function that is based on an
    *expression template* given to it. The expression template describes a
    function of `solve_component`.

    Furthermore, the code generator or interpreter calls `ScipySolverGenerator`
    with each :class:`AssignSolved` instruction with the corresponding solver
    tag. `ScipySolverGenerator` takes the expression to be solved for (the
    "concrete expression"), matches it with the expression template, and then
    returns a pymbolic expression with the appropriate code to call the solver
    function.

    """

    def __init__(self, expression_template, solve_component, solver_id="solver",
                 function_registry=base_function_registry):
        """
        :arg expression_template: The expression template as a pymbolic expression
        :arg solve_component: The name of the variable to be solved for
        :arg function_registry: The `class`:FunctionRegistry to be used for
                                generating code in the solver function
        """
        from leap.vm.utils import get_variables
        variables = get_variables(expression_template,
                                  include_function_symbols=True)

        if solve_component not in variables:
            raise ValueError("The solve component must be in the expression")

        self.expression = expression_template
        self.solve_component = solve_component
        self.solver_id = solver_id
        self.guess = "<arg>guess"

        self.solver_func = var("<func>" + solver_id)

        self._infer_args(variables - set([solve_component]), function_registry)
        self.args = (self.guess,) + self.global_args + self.free_args

        self._make_fake_name_manager()

        unregistered_functions = set(global_ for global_ in self.global_args
                                     if global_.startswith("<func>"))
        self._augment_function_registry(unregistered_functions, function_registry)

    def _infer_args(self, variables, function_registry):
        # Remove registered functions from the potential arguments - the codegen
        # object in the function registry is responsible for making the calls.
        registered_functions = set(var for var in variables
                                   if var in function_registry)
        variables -= registered_functions

        # Find global arguments.
        from leap.vm.utils import is_state_variable
        is_global = lambda var: is_state_variable(var) or var.startswith("<func>")
        self.global_args = tuple(var for var in sorted(variables) if is_global(var))

        # Find free arguments.
        variables -= set(self.global_args)
        self.free_args = tuple(sorted(variables))

    def _make_fake_name_manager(self):
        from leap.vm.codegen.utils import KeyToUniqueNameMap

        name_map = KeyToUniqueNameMap(
            start={self.solve_component: "var_x",
                   self.guess: "var_guess"},
            forced_prefix="var_")

        class KeyToUniqueNameMapWrapper:

            def __init__(self, name_map):
                self.name_map = name_map

            def __getitem__(self, key):
                return name_map.get_or_make_name_for_key(key)

            name_function = __getitem__

        self.name_manager = KeyToUniqueNameMapWrapper(name_map)

    def _augment_function_registry(self, unregistered_functions, function_registry):
        """
        By default, PythonExpressionMapper will generate function calls in a way
        that is unfit to be used outside of a method class, unless overriden by
        the function registry. In order to avoid having the user need to supply
        a function registry to generate correct calling code, we provide a
        default code generator.
        """

        from leap.vm.function_registry import Function

        class PythonCallGenerator(object):

            def __init__(self, function_name):
                self.function_name = function_name

            def __call__(self, expr_mapper, arg_strs_dict):
                pos_args = []
                kw_args = []
                for index, val in arg_strs_dict.items():
                    if isinstance(index, str):
                        kw_args.append(index + "=" + val)
                    elif isinstance(index, int):
                        pos_args.append((index, val))
                from operator import itemgetter
                pos_args = [pair[1] for pair in sorted(pos_args, key=itemgetter(0))]
                return "{name}({args})".format(name=self.function_name,
                                               args=', '.join(pos_args + kw_args))

        for unregistered_function in unregistered_functions:
            function = Function(identifier=unregistered_function,
                                arg_names=()).register_codegen(
                                    "python",
                                    PythonCallGenerator(
                                        self.name_manager[unregistered_function]))
            function_registry = function_registry.register(function)

        self.function_registry = function_registry

    def __call__(self, expression, solve_component, guess):
        """Callback provided for the interpreter or code generator to generate an
        expression from an :class:`AssignSolved` instance.

        The following are the conventions used to match the expression template
        to a concrete expression:

         * functions and state variables (eg. `<func>...`, `<dt>`, `<state>...`)
           in the expression template are not matched, they are left alone;
         * the solve component in the template is matched to the solve component
           in the concrete expression;
         * the remaining variables in the expression template are matched
           to subexpressions in the concrete expression.

        :arg expression: The concrete expression as a pymbolic expression
        :arg solve_component: The name of the solve component in the concrete
                              expression
        :arg guess: A pymbolic expression for the guess
        """

        from leap.vm.expression import match, substitute
        # Rename the solve component in self.expression to solve_component.
        template = substitute(self.expression,
                              {self.solve_component: solve_component})
        subst = match(template, expression, self.free_args)

        if set(self.free_args) < set(subst.keys()):
            raise ValueError("Cannot match entire template with " + str(expression))

        return self.solver_func(guess,
            *(tuple(var(arg) for arg in self.global_args) +
              tuple(subst[arg] for arg in self.free_args)))

    def get_solver_code(self, function_name, function_registry):
        """Create and return the code for the function called by the solver
        expressions.

        :arg function_identifier: The identifier for the function to be created
        :arg function_registry: The function registry to use for emitting code
        """

        from pytools.py_codegen import PythonFunctionGenerator, Indentation
        from leap.vm.codegen.expressions import PythonExpressionMapper

        mapper = PythonExpressionMapper(self.name_manager, self.function_registry)

        args = (mapper(var(arg)) for arg in self.args)
        emit = PythonFunctionGenerator(function_name, args)

        emit("import numpy")
        emit("import scipy.optimize")
        emit("def f(var_x):")
        with Indentation(emit):
            emit("return " + mapper(self.expression))
        emit("if not numpy.isscalar(var_guess):")
        with Indentation(emit):
            emit("return scipy.optimize.root(f, " + mapper(var(self.guess)) + ").x")
        emit("else:")
        with Indentation(emit):
            emit("return scipy.optimize.newton(f, " + mapper(var(self.guess)) + ")")
        return emit.get()

    def get_compiled_solver(self, function_name=None,
                            function_registry=base_function_registry):
        from leap.vm.codegen.utils import exec_in_new_namespace, \
            make_identifier_from_name
        name = function_name or make_identifier_from_name(self.solver_id)
        code = self.get_solver_code(name, function_name)
        return exec_in_new_namespace(code)[name]


def replace_AssignSolved(dag, *solver_hooks):
    """
    :arg dag: The TimeIntegratorCode instance
    :arg solver_hooks: A map from solver names to expression generators
    """

    new_instructions = []

    from leap.vm.language import AssignExpression, AssignSolved

    for insn in dag.instructions:

        if not isinstance(insn, AssignSolved):
            new_instructions.append(insn)
            continue

        expression = insn.expression
        solve_component = insn.solve_component.name
        guess = insn.guess

        solver = solver_hooks[insn.solver_id]

        new_instructions.append(
            AssignExpression(assignee=insn.assignee,
                             expression=solver(expression, solve_component, guess),
                             id=insn.id,
                             depends_on=insn.depends_on))

    return dag.copy(instructions=new_instructions)
