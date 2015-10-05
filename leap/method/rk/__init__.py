"""Runge-Kutta ODE timestepper."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007-2013 Andreas Kloeckner
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

import numpy as np
from leap.method import Method, TwoOrderAdaptiveMethod
from leap.vm.language import CodeBuilder, TimeIntegratorCode

from pymbolic import var


__doc__ = """
.. autoclass:: ODE23Method

.. autoclass:: ODE45Method

.. autoclass:: MidpointMethod
.. autoclass:: HeunsMethod
.. autoclass:: RK4Method

.. autoclass:: LSRK4Method
"""


# {{{ utilities

def _truncate_final_zeros(array):
    if not array:
        return array

    index = len(array) - 1
    while array[index] == 0 and index >= 0:
        index -= 1
    return array[:index + 1]


def _is_first_stage_same_as_last_stage(c, coeff_set):
    return (
            c
            and c[0] == 0
            and c[-1] == 1
            and not _truncate_final_zeros(coeff_set[0]))


def _is_last_stage_same_as_output(c, coeff_sets, output_stage_coefficients):
    return (
            c

            and all(
                coeff_set[-1]
                for coeff_set in coeff_sets)

            and output_stage_coefficients

            and all(
                _truncate_final_zeros(coeff_set[-1])
                ==
                _truncate_final_zeros(output_stage_coefficients)
                for coeff_set in coeff_sets))

# }}}


# {{{ fully general butcher tableau to code

class ButcherTableauMethod(Method):
    """Explicit and implicit Runge-Kutta methods."""

    def __init__(self, component_id, limiter_name=None):

        self.component_id = component_id

        self.dt = var('<dt>')
        self.t = var('<t>')
        self.state = var('<state>' + component_id)

        self.limiter_name = limiter_name

        if self.limiter_name is not None:
            self.limiter = var("<func>" + self.limiter_name)
        else:
            self.limiter = None

    def generate_butcher(self, stage_coeff_set_names, stage_coeff_sets, rhs_funcs,
            estimate_coeff_set_names, estimate_coeff_sets):
        """
        :arg stage_coeff_set_names: a list of names/string identifiers
            for stage coefficient sets
        :arg stage_coeff_sets: a mapping from set names to stage coefficients
        :arg rhs_funcs: a mapping from set names to right-hand-side
            functions
        :arg estimate_coeffs_set_names: a list of names/string identifiers
            for estimate coefficient sets
        :arg estimate_coeffs_sets: a mapping from estimate coefficient set
            names to cofficients.
        """

        from pymbolic import var
        comp = self.component_id

        dt = self.dt
        t = self.t
        state = self.state

        nstages = len(self.c)

        # {{{ check coefficients for plausibility

        for name in stage_coeff_set_names:
            for istage in range(nstages):
                coeff_sum = sum(stage_coeff_sets[name][istage])
                assert abs(coeff_sum - self.c[istage]) < 1e-12, (
                        name, istage, coeff_sum, self.c[istage])
        # }}}

        # {{{ initialization

        last_rhss = {}

        with CodeBuilder(label="initialization") as cb:
            for name in stage_coeff_set_names:
                if (
                        name in self.recycle_last_stage_coeff_set_names
                        and _is_first_stage_same_as_last_stage(
                        self.c, stage_coeff_sets[name])):
                    last_rhss[name] = var("<p>last_rhs_" + name)
                    cb(last_rhss[name], rhs_funcs[name](t=t, **{comp: state}))

        cb_init = cb

        # }}}

        stage_rhs_vars = {}
        rhs_var_to_unknown = {}
        for name in stage_coeff_set_names:
            stage_rhs_vars[name] = [
                    cb.fresh_var('rhs_%s_s%d' % (name, i)) for i in range(nstages)]

            # These are rhss if they are not yet known and pending an implicit solve.
            for i, rhsvar in enumerate(stage_rhs_vars[name]):
                unkvar = cb.fresh_var('unk_%s_s%d' % (name, i))
                rhs_var_to_unknown[rhsvar] = unkvar

        knowns = set()

        # {{{ stage loop

        with CodeBuilder(label="primary") as cb:
            equations = []
            unknowns = set()

            def make_known(v):
                unknowns.discard(v)
                knowns.add(v)

            for istage in range(nstages):
                for name in stage_coeff_set_names:
                    c = self.c[istage]
                    my_rhs = stage_rhs_vars[name][istage]

                    if (
                            name in self.recycle_last_stage_coeff_set_names
                            and istage == 0
                            and _is_first_stage_same_as_last_stage(
                                self.c, stage_coeff_sets[name])):
                        cb(my_rhs, last_rhss[name])
                        make_known(my_rhs)

                    else:
                        is_implicit = False

                        state_increment = 0
                        for src_name in stage_coeff_set_names:
                            coeffs = stage_coeff_sets[src_name][istage]
                            for src_istage, coeff in enumerate(coeffs):
                                rhsval = stage_rhs_vars[src_name][src_istage]
                                if rhsval not in knowns:
                                    unknowns.add(rhsval)
                                    is_implicit = True

                                state_increment += dt * coeff * rhsval

                        state_est = state + state_increment
                        rhs_expr = rhs_funcs[name](t=t + c*dt, **{comp: state_est})

                        if is_implicit:
                            from leap.vm.expression import collapse_constants
                            solve_expression = collapse_constants(
                                    my_rhs - rhs_expr,
                                    list(unknowns) + [self.state],
                                    cb.assign, cb.fresh_var)
                            equations.append(solve_expression)

                        else:
                            if self.limiter is not None:
                                rhs_expr = self.limiter(rhs_expr)
                            cb(my_rhs, rhs_expr)
                            make_known(my_rhs)

                    # {{{ emit solve if possible

                    if unknowns and len(unknowns) == len(equations):
                        # got a square system, let's solve
                        if self.limiter is not None:
                            temp_vars = [
                                    cb.fresh_var(unk.name + "_pre_lim")
                                    for unk in unknowns]

                            assignees = [tv.name for tv in temp_vars]
                        else:
                            assignees = [unk.name for unk in unknowns]

                        from leap.vm.expression import substitute
                        subst_dict = dict(
                                (rhs_var.name, rhs_var_to_unknown[rhs_var].name)
                                for rhs_var in unknowns)

                        cb.assign_solved(
                                assignees=assignees,
                                solve_components=[
                                    rhs_var_to_unknown[unk].name
                                    for unk in unknowns],
                                expressions=[
                                    substitute(eq, subst_dict)
                                    for eq in equations],

                                # TODO: Could supply a starting guess
                                other_params={
                                    "guess": state},
                                solver_id="solve")

                        if self.limiter is not None:
                            for unk, tv in zip(unknowns, temp_vars):
                                cb(unk, tv)

                        del equations[:]
                        knowns.update(unknowns)
                        unknowns.clear()

                    # }}}

            # Compute solution estimates.
            estimate_vars = [
                    cb.fresh_var("est_"+name)
                    for name in estimate_coeff_set_names]

            for iest, name in enumerate(estimate_coeff_set_names):
                out_coeffs = estimate_coeff_sets[name]

                if _is_last_stage_same_as_output(self.c,
                        stage_coeff_sets, out_coeffs):
                    cb(
                            estimate_vars,
                            stage_rhs_vars[stage_coeff_set_names[-1]][-1])

                else:
                    state_increment = 0
                    for src_name in stage_coeff_set_names:
                        state_increment += sum(
                                    coeff * stage_rhs_vars[src_name][src_istage]
                                    for src_istage, coeff in enumerate(out_coeffs))

                    cb(
                            estimate_vars[iest],
                            state + dt*state_increment)

            cb.fence()

            self.finish(cb, estimate_coeff_set_names, estimate_vars)

            # These updates have to happen *after* finish because before we
            # don't yet know whether finish will accept the new state.

            for name in stage_coeff_set_names:
                if (
                        name in self.recycle_last_stage_coeff_set_names
                        and _is_first_stage_same_as_last_stage(
                            self.c, stage_coeff_sets[name])):
                    cb(last_rhss[name], stage_rhs_vars[name][-1])

            cb.fence()
            cb(self.t, self.t + self.dt)

        cb_primary = cb

        # }}}

        return TimeIntegratorCode.create_with_init_and_step(
            instructions=cb_init.instructions | cb_primary.instructions,
            initialization_dep_on=cb_init.state_dependencies,
            step_dep_on=cb_primary.state_dependencies)

    def finish(self, cb, estimate_names, estimate_vars):
        cb(self.state, estimate_vars[0])
        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')

# }}}


# {{{ simple butcher tableau methods

class SimpleButcherTableauMethod(ButcherTableauMethod):
    def generate(self):
        return self.generate_butcher(
                stage_coeff_set_names=("explicit",),
                stage_coeff_sets={
                    "explicit": self.a_explicit},
                rhs_funcs={"explicit": var("<func>"+self.component_id)},
                estimate_coeff_set_names=("main",),
                estimate_coeff_sets={
                    "main": self.output_coeffs,
                    })


class MidpointMethod(SimpleButcherTableauMethod):
    c = [0, 1/2]

    a_explicit = (
            (),
            (1/2,),
            )

    output_coeffs = (0, 1)

    recycle_last_stage_coeff_set_names = ()


class HeunsMethod(SimpleButcherTableauMethod):
    c = [0, 1]

    a_explicit = (
            (),
            (1,),
            )

    output_coeffs = (1/2, 1/2)

    recycle_last_stage_coeff_set_names = ()


class RK4Method(SimpleButcherTableauMethod):
    c = (0, 1/2, 1/2, 1)

    a_explicit = (
            (),
            (1/2,),
            (0, 1/2,),
            (0, 0, 1,),
            )

    output_coeffs = (1/6, 1/3, 1/3, 1/6)

    recycle_last_stage_coeff_set_names = ()

# }}}


# {{{ Embedded Runge-Kutta schemes base class

class EmbeddedButcherTableauMethod(ButcherTableauMethod, TwoOrderAdaptiveMethod):
    """
    User-supplied context:
        <state> + component_id: The value that is integrated
        <func> + component_id: The right hand side function
    """

    def __init__(self, component_id, use_high_order=True, limiter_name=None,
            atol=0, rtol=0, max_dt_growth=None, min_dt_shrinkage=None):
        ButcherTableauMethod.__init__(
                self,
                component_id=component_id,
                limiter_name=limiter_name)

        TwoOrderAdaptiveMethod.__init__(
                self,
                atol=atol,
                rtol=rtol,
                max_dt_growth=max_dt_growth,
                min_dt_shrinkage=min_dt_shrinkage)

        self.use_high_order = use_high_order

    def generate(self):
        if self.use_high_order:
            estimate_names = ("high_order", "low_order")
        else:
            estimate_names = ("low_order", "high_order")

        return self.generate_butcher(
                stage_coeff_set_names=("explicit",),
                stage_coeff_sets={
                    "explicit": self.a_explicit},
                rhs_funcs={"explicit": var("<func>"+self.component_id)},
                estimate_coeff_set_names=estimate_names,
                estimate_coeff_sets={
                    "high_order": self.high_order_coeffs,
                    "low_order": self.low_order_coeffs
                    })

    def finish(self, cb, estimate_coeff_set_names, estimate_vars):
        if not self.adaptive:
            super(EmbeddedButcherTableauMethod, self).finish(
                    cb, estimate_coeff_set_names, estimate_vars)
        else:
            high_est = estimate_vars[
                    estimate_coeff_set_names.index("high_order")]
            low_est = estimate_vars[
                    estimate_coeff_set_names.index("low_order")]
            self.finish_adaptive(cb, high_est, low_est)

    def finish_nonadaptive(self, cb, high_order_estimate, low_order_estimate):
        if self.use_high_order:
            est = high_order_estimate
        else:
            est = low_order_estimate

        cb(self.state, est)
        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')

# }}}


# {{{ Bogacki-Shampine second/third-order Runge-Kutta

class ODE23Method(EmbeddedButcherTableauMethod):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    http://dx.doi.org/10.1016/0893-9659(89)90079-7
    """

    c = [0, 1/2, 3/4, 1]

    a_explicit = [
            [],
            [1/2],
            [0, 3/4],
            [2/9, 1/3, 4/9],
            ]

    low_order = 2
    low_order_coeffs = [7/24, 1/4, 1/3, 1/8]
    high_order = 3
    high_order_coeffs = [2/9, 1/3, 4/9, 0]

    recycle_last_stage_coeff_set_names = ("explicit",)

# }}}


# {{{ Dormand-Prince fourth/fifth-order Runge-Kutta

class ODE45Method(EmbeddedButcherTableauMethod):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    http://dx.doi.org/10.1016/0771-050X(80)90013-3.
    """

    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a_explicit = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
            ]

    low_order = 4
    low_order_coeffs = [5179/57600, 0, 7571/16695, 393/640, -92097/339200,
            187/2100, 1/40]
    high_order = 5
    high_order_coeffs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

    recycle_last_stage_coeff_set_names = ("explicit",)

# }}}


# {{{ Carpenter/Kennedy low-storage fourth-order Runge-Kutta

class LSRK4Method(Method):
    """A low storage fourth-order Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.64
    or
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994
    """

    _RK4A = [
            0.0,
            -567301805773 / 1357537059087,
            -2404267990393 / 2016746695238,
            -3550918686646 / 2091501179385,
            -1275806237668 / 842570457699,
            ]

    _RK4B = [
            1432997174477 / 9575080441755,
            5161836677717 / 13612068292357,
            1720146321549 / 2090206949498,
            3134564353537 / 4481467310338,
            2277821191437 / 14882151754819,
            ]

    _RK4C = [
            0.0,
            1432997174477/9575080441755,
            2526269341429/6820363962896,
            2006345519317/3224310063776,
            2802321613138/2924317926251,
            #1,
            ]
    coeffs = np.array([_RK4A, _RK4B, _RK4C]).T

    adaptive = False

    def __init__(self, component_id, limiter_name=None):
        """
        :arg component_id: an identifier to be used for the single state
            component supported.
        """

        # Set up variables.
        from pymbolic import var

        self.component_id = component_id

        if limiter_name is not None:
            self.limiter = var("<func>" + self.limiter_name)
        else:
            self.limiter = None

    def generate(self):
        comp_id = self.component_id

        from pymbolic import var
        dt = var("<dt>")
        t = var("<t>")
        residual = var("<p>residual_" + comp_id)
        state = var("<state>" + comp_id)
        rhs_func = var("<func>" + comp_id)

        with CodeBuilder("initialization") as cb:
            cb(residual, 0)

        cb_init = cb

        # Primary.

        rhs_val = var("rhs_val")

        with CodeBuilder("primary") as cb:
            for a, b, c in self.coeffs:
                cb.fence()
                cb(rhs_val, rhs_func(t=t + c*dt, **{comp_id: state}))
                cb(residual, a*residual + dt*rhs_val)
                new_state_expr = state + b * residual

                if self.limiter is not None:
                    new_state_expr = self.limiter(**{comp_id: new_state_expr})

                cb.fence()
                cb(state, new_state_expr)

            cb.yield_state(state, comp_id, t + dt, 'final')
            cb.fence()
            cb(t, t + dt)

        cb_primary = cb

        from leap.vm.language import TimeIntegratorCode
        return TimeIntegratorCode.create_with_init_and_step(
            instructions=cb_init.instructions | cb_primary.instructions,
            initialization_dep_on=cb_init.state_dependencies,
            step_dep_on=cb_primary.state_dependencies)

# }}}


# vim: foldmethod=marker
