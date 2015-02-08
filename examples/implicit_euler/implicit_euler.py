"""Implicit Euler timestepper"""

from __future__ import division

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

from leap.method import Method
from leap.vm.language import TimeIntegratorCode, NewCodeBuilder
from pymbolic import var
from pymbolic.primitives import CallWithKwargs


class ImplicitEulerMethod(Method):
    """
    Context:
       state: The value that is integrated
       rhs_func: The right hand side function
    """

    def __init__(self, component_id):
        self.component_id = component_id
        self.dt = var('<dt>')
        self.t = var('<t>')
        self.state = var('<state>' + component_id)
        self.rhs_func = var('<func>' + component_id)

    def generate(self, solver_hook):
        """Return code that implements the implicit Euler method for the single
        state component supported."""

        with NewCodeBuilder(label="primary") as cb:
            self._make_primary(cb)

        code = TimeIntegratorCode.create_with_steady_state(
            dep_on=cb.state_dependencies,
            instructions=cb.instructions)

        from leap.vm.implicit import replace_AssignSolved

        return replace_AssignSolved(code, solver_hook)

    def _make_primary(self, builder):
        """Add code to drive the primary stage."""

        solve_component = var('next_state')
        solve_expression = solve_component - self.state - \
                           self.dt * CallWithKwargs(
                               function=self.rhs_func,
                               parameters=(),
                               kw_parameters={
                                   't': self.t + self.dt,
                                   self.component_id: solve_component
                               })

        builder.assign_solved(self.state, solve_component,
                              solve_expression, self.state, 0)

        builder.yield_state(self.state, self.component_id,
                            self.t + self.dt, 'final')
        builder.fence()
        builder.assign(self.t, self.t + self.dt)

    def implicit_expression(self, expression_tag=None):
        from leap.vm.expression import parse
        return (parse("`solve_component` + state + dt * `{rhs}`(t=t,"
                      "{component_id}=`solve_component`)".format(
                          rhs=self.rhs_func.name,
                          component_id=self.component_id)),
                "solve_component")
