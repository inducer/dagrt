# -*- coding: utf-8 -*-

"""Auxiliary routines used by Adams-Bashforth ODE timesteppers."""

from __future__ import division


__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
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

import numpy
import numpy.linalg as la
import six.moves


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
    return make_generic_ab_coefficients(numpy.arange(0, -order, -1), 0, 1)


def make_implicit_ab_coefficients(order):
    return make_generic_ab_coefficients(numpy.arange(1, -order + 1, -1), 0, 1)


def linear_comb(coefficients, vectors):
    from operator import add
    return six.moves.reduce(add,
            (coeff * v for coeff, v in
                zip(coefficients, vectors)))
