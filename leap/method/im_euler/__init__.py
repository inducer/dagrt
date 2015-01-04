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

    def __call__(self, component_id):
        """Return code that implements the implicit Euler method for the single
        state component supported."""

        self._dt = var('<dt>')
        self._t = var('<t>')
        self._state = var('<state>' + component_id)
        self._rhs = var('<func>' + component_id)
        self._component_id = component_id

        with NewCodeBuilder(label="primary") as cb:
            self._make_primary(cb)

        return TimeIntegratorCode.create_with_steady_state(
            dep_on=cb.state_dependencies,
            instructions=cb.instructions)

    def _make_primary(self, builder):
        """Add code to drive the primary stage."""
        from leap.vm.expression import collapse_constants

        solve_component = var('next_state')
        solve_expression = collapse_constants(
            solve_component - self._state - self._dt *
            CallWithKwargs(
                function=self._rhs,
                parameters=(),
                kw_parameters={
                    't': self._t + self._dt,
                    self._component_id: solve_component
                    }),
            [solve_component],
            builder.assign,
            builder.fresh_var)

        builder.fence()            
        builder.assign_solved(self._state, solve_component,
                              solve_expression, self._state,
                              'newton')

        builder.yield_state(self._state, self._component_id,
                            self._t + self._dt, 'final')
        builder.fence()
        builder.assign(self._t, self._t + self._dt)
