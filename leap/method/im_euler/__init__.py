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
from leap.vm.language import (AssignExpression, AssignSolvedRHS, ReturnState,
                              TimeIntegratorCode, CodeBuilder)
from pymbolic import var
from pymbolic.primitives import CallWithKwargs


class ImplicitEulerMethod(Method):

    def __call__(self, component_id):
        """Return code that implements the implicit Euler method for the single
        state component supported."""

        self._dt = var('<dt>')
        self._t = var('<t>')
        self._state = var('<state>' + component_id)
        self._component_id = component_id

        cbuild = CodeBuilder()

        self._make_primary(cbuild)

        return TimeIntegratorCode(
            instructions=cbuild.instructions,
            initialization_dep_on=frozenset(),
            step_dep_on=['return', 'increment_t'],
            step_before_fail=True)

    def _make_primary(self, cbuild):
        """Add code to drive the primary stage."""

        cbuild.add_and_get_ids(
            AssignSolvedRHS(
                assignee=self._state.name,
                solve_component=var('next_state'),
                t=self._t + self._dt,
                lhs=[
                    var('next_state') - self._state -
                    self._dt * CallWithKwargs(
                        function=var(self._component_id),
                        parameters=(),
                        kw_parameters={
                            't': self._t + self._dt,
                            self._component_id: var('next_state')
                        })
                    ],
                rhs=[0],
                guess=self._state,
                solver_id='newton',
                id='step'),
            ReturnState(
                time_id='final',
                time=self._t + self._dt,
                component_id=self._component_id,
                expression=self._state,
                depends_on=['step'],
                id='return'),
            AssignExpression(
                assignee=self._t.name,
                expression=self._t + self._dt,
                depends_on=['step', 'return'],
                id='increment_t'))

        cbuild.commit()
