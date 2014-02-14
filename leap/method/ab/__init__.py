#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Adams-Bashforth ODE solvers."""

from __future__ import division

__copyright__ = 'Copyright (C) 2007 Andreas Kloeckner'

__license__ = \
    """
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

import numpy
import numpy.linalg as la
from leap.method import Method


def generic_vandermonde(points, functions):
    """Return a Vandermonde matrix.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and points is
    the list of :math:`x_i`.
    """

    v = numpy.zeros((len(points), len(functions)))
    for (i, x) in enumerate(points):
        for (j, f) in enumerate(functions):
            v[i, j] = f(x)
    return v


def monomial_vdm(points, max_degree=None):

    class Monomial:

        def __init__(self, expt):
            self.expt = expt

        def __call__(self, x):
            return x ** self.expt

    if max_degree is None:
        max_degree = len(points) - 1

    return generic_vandermonde(points, [Monomial(i) for i in
                               range(max_degree + 1)])


# coefficient generators ------------------------------------------------------

def make_generic_ab_coefficients(levels, int_start, tap):
    """Find coefficients (αᵢ) such that
       ∑ᵢ αᵢ F(tᵢ) = ∫[int_start..tap] f(t) dt."""

    # explanations --------------------------------------------------------------
    # To calculate the AB coefficients this method makes use of the interpolation
    # connection of the Vandermonde matrix:
    #
    #  Vᵀ * α = fe(t₀₊₁),                                    (1)
    #
    # with Vᵀ as the transposed Vandermonde matrix (with monomial base: xⁿ),
    #
    #  α = (..., α₋₂, α₋₁,α₀)ᵀ                               (2)
    #
    # a vector of interpolation coefficients and
    #
    #  fe(t₀₊₁) = (t₀₊₁⁰, t₀₊₁¹, t₀₊₁²,...,t₀₊₁ⁿ)ᵀ           (3)
    #
    # a vector of the evaluated interpolation polynomial f(t) at t₀₊₁ = t₀ ∓ h
    # (h being any arbitrary stepsize).
    #
    # Solving the system (1) by knowing Vᵀ and fe(t₀₊₁) receiving α makes it
    # possible for any function F(t) - the function which gets interpolated
    # by the interpolation polynomial f(t) - to calculate f(t₀₊₁) by:
    #
    # f(t₀₊₁) =  ∑ᵢ αᵢ F(tᵢ)                                 (5)
    #
    # with F(tᵢ) being the values of F(t) at the sampling points tᵢ.
    # --------------------------------------------------------------------------
    # The Adams-Bashforth method is defined by:
    #
    #  y(t₀₊₁) = y(t₀) + Δt * ∫₀⁰⁺¹ f(t) dt                  (6)
    #
    # with:
    #
    #  ∫₀⁰⁺¹ f(t) dt = ∑ᵢ ABcᵢ F(tᵢ),                        (8)
    #
    # with ABcᵢ = [AB coefficients], f(t) being the interpolation polynomial,
    # and F(tᵢ) being the values of F (= RHS) at the sampling points tᵢ.
    # --------------------------------------------------------------------------
    # For the AB method (1) becomes:
    #
    #  Vᵀ * ABc = ∫₀⁰⁺¹ fe(t₀₊₁)                             (7)
    #
    # with ∫₀⁰⁺¹ fe(t₀₊₁) being a vector evalueting the integral of the
    # interpolation polynomial in the form oft
    #
    #  1/(n+1)*(t₀₊₁⁽ⁿ⁾-t₀⁽ⁿ⁾)                               (8)
    #
    #  for n = 0,1,...,N sampling points, and
    #
    # ABc = [c₀,c₁, ... , cn]ᵀ                               (9)
    #
    # being the AB coefficients.
    #
    # For example ∫₀⁰⁺¹ f(t₀₊₁) evaluated for the timestep [t₀,t₀₊₁] = [0,1]
    # is:
    #
    #  point_eval_vec = [1, 0.5, 0.333, 0.25, ... ,1/n]ᵀ.
    #
    # For substep levels the bounds of the integral has to be adapted to the
    # size and position of the substep interval:
    #
    #  [t₀,t₀₊₁] = [substep_int_start, substep_int_end]
    #
    # which is equal to the implemented [int_start, tap].
    #
    # Since Vᵀ and ∫₀⁰⁺¹ f(t₀₊₁) is known the AB coefficients c can be
    # predicted by solving system (7) and calculating:
    #
    #  ∫₀⁰⁺¹ f(t) dt = ∑ᵢ ABcᵢ F(tᵢ),

    # from hedge.polynomial import monomial_vdm

    point_eval_vec = numpy.array([1 / (n + 1) * (tap ** (n + 1)
                                 - int_start ** (n + 1)) for n in
                                 range(len(levels))])
    return la.solve(monomial_vdm(levels).T, point_eval_vec)


def make_ab_coefficients(order):
    return make_generic_ab_coefficients(numpy.arange(0, -order, -1), 0,
            1)


# time steppers ---------------------------------------------------------------

class AdamsBashforthTimeStepper(Method):

    def __init__(
        self,
        order,
        dtype=numpy.float64,
        rcon=None,
        ):
        from pytools import match_precision
        self.dtype = numpy.dtype(dtype)
        self.scalar_dtype = match_precision(numpy.dtype(numpy.float64),
                self.dtype)
        self.coeffs = numpy.asarray(make_ab_coefficients(order),
                dtype=self.scalar_dtype)[::-1]
        from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper
        if order <= 2:
            self.rk_tableau = ODE23TimeStepper.butcher_tableau
            self.rk_coeffs = ODE23TimeStepper.low_order_coeffs
        elif order == 3:
            self.rk_tableau = ODE23TimeStepper.butcher_tableau
            self.rk_coeffs = ODE23TimeStepper.high_order_coeffs
        elif order == 4:
            self.rk_tableau = ODE45TimeStepper.butcher_tableau
            self.rk_coeffs = ODE45TimeStepper.low_order_coeffs
        elif order == 5:
            self.rk_tableau = ODE45TimeStepper.butcher_tableau
            self.rk_coeffs = ODE45TimeStepper.high_order_coeffs
        else:
            raise ValueError('Unsupported order: %s' % order)

    @property
    def order(self):
        return len(self.coeffs)

    def bootstrap(self, cbuild, component_id):
        """Initialize the timestepper with an RK method."""

        from leap.vm.language import AssignRHS, AssignExpression, If

        add_and_get_ids = cbuild.add_and_get_ids
        from pymbolic import var

        step = var('<p>step')
        fvals = [var('<p>last_rhs_%d' % i) for i in
                 xrange(self.order - 1, 0, -1)]
        state = var('<state>' + component_id)
        t = var('<t>')
        dt = var('<dt>')
        last_rhs = var('<p>last_rhs_' + component_id)
        
        # Save the current RHS to the AB history
        
        condition_ids = []
        
        rhs = var("rhs")
        
        compute_rhs_id, = \
            add_and_get_ids(AssignRHS(assignees=(rhs.name,),
                            component_id=component_id, t=t,
                            rhs_arguments=(((component_id, state),),),))

        for (i, fval) in enumerate(fvals):
            from pymbolic.primitives import Comparison
            assign_id, = add_and_get_ids(AssignExpression(fval.name,
                    rhs, depends_on=[compute_rhs_id]))
            condition_id, = \
                add_and_get_ids(If(condition=Comparison(step, '==', i),
                                then_depends_on=[assign_id],
                                else_depends_on=[]))
            condition_ids.append(condition_id)
            
        # Compute the new value of the state
        # Stage loop (taken from EmbeddedButcherTableauMethod)

        rhss = []

        all_rhs_eval_ids = []

        for (istage, (c, coeffs)) in enumerate(self.rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                this_rhs = last_rhs
            else:
                stage_state = state + sum(dt * coeff * rhss[j] for (j,
                        coeff) in enumerate(coeffs))

                rhs_id = 'rhs%d' % istage
                rhs_insn_id = 'ev_rhs%d' % istage
                all_rhs_eval_ids.append(rhs_insn_id)

                add_and_get_ids(AssignRHS(assignees=(rhs_id, ),
                                component_id=component_id, t=t + c
                                * dt, rhs_arguments=(((component_id,
                                stage_state), ), ), id=rhs_insn_id))

                this_rhs = var(rhs_id)
            rhss.append(this_rhs)

        add_and_get_ids(AssignExpression(state.name, state + sum(dt
                        * coeff * rhss[j] for (j, coeff) in
                        enumerate(self.rk_coeffs)), id='rk_update_state'
                        , depends_on=all_rhs_eval_ids + condition_ids))
        
        last_rhs_assignment_id, = \
            add_and_get_ids(AssignExpression(last_rhs.name, this_rhs,
                            depends_on=['rk_update_state']))

        return [last_rhs_assignment_id, 'rk_update_state']

    def __call__(self, component_id):
        from leap.vm.language import AssignRHS, AssignNorm, \
            AssignExpression, ReturnState, If, Raise, FailStep, \
            TimeIntegratorCode, CodeBuilder

        cbuild = CodeBuilder()
        add_and_get_ids = cbuild.add_and_get_ids

        from pymbolic import var

        # Declare variables

        step = var('<p>step')
        fvals = [var('<p>last_rhs_%d' % i) for i in
                 xrange(self.order - 1, 0, -1)]
        state = var('<state>' + component_id)
        t = var('<t>')
        dt = var('<dt>')
        last_rhs = var('<p>last_rhs_' + component_id)
        curr_rhs = var('curr_rhs')

        dep_inf_exclude_names = ['<t>', '<dt>', state.name, last_rhs.name,
            step.name]

        # Initialize variables

        initialization_dep_on = \
            add_and_get_ids(AssignExpression(step.name, 0),
                            AssignRHS(assignees=(last_rhs.name, ),
                            component_id=component_id, t=t,
                            rhs_arguments=(((component_id, state), ),
                            )))

        cbuild.commit()

        # RK bootstrap stage

        bootstrap_ids = self.bootstrap(cbuild, component_id)
        
        add_and_get_ids(AssignExpression(step.name, step + 1,
                        id='increment_step'))

        cbuild.infer_single_writer_dependencies(exclude=dep_inf_exclude_names)
        cbuild.commit()
        
        # AB stage
                
        add_and_get_ids(
            AssignRHS(assignees=(curr_rhs.name,), component_id=component_id,
                      t=t, rhs_arguments=(((component_id, state),),),
                      id='compute_curr_rhs'),
            AssignExpression(state.name, state + dt *
                (sum(self.coeffs[i] * fvals[i] for i in xrange(0, self.order-1))
                 + curr_rhs * self.coeffs[-1]), id='ab_update_state',
                depends_on=['compute_curr_rhs'])
            )
        
        cbuild.commit()
        
        # Update AB history
        
        last_dep_id = 'ab_update_state'
        
        for i, fval in enumerate(fvals):
            next_fval = fvals[i + 1] if i + 1 < len(fvals) else curr_rhs
            last_dep_id, = add_and_get_ids(AssignExpression(fval.name,
                            next_fval, depends_on=[last_dep_id]))
                
        # The branch to decide whether the current step is an initialization
        # step or an AB timestepping step
        
        from pymbolic.primitives import Comparison

        main_branch_id, = \
            add_and_get_ids(If(condition=Comparison(step, '<',
                            self.order - 1), then_depends_on=bootstrap_ids +
                            ['increment_step'], else_depends_on=[last_dep_id]))
        
        # Increment t and return the state

        add_and_get_ids(AssignExpression(t.name, t + dt,
                        id='increment_t', depends_on=[main_branch_id]),
            ReturnState(
            id='ret_state',
            time_id='final',
            time=t + dt,
            component_id=component_id,
            expression=state,
            depends_on=[main_branch_id],
            ))
        
        cbuild.commit()

        return TimeIntegratorCode(instructions=cbuild.instructions,
                                  initialization_dep_on=initialization_dep_on,
                                  step_dep_on=['ret_state', 'increment_t'],
                                  step_before_fail=False)


