"""Kennedy-Carpenter implicit/explicit RK."""

from __future__ import division

__copyright__ = """
Copyright (C) 2010, 2013 Andreas Kloeckner
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


from leap.method.rk import EmbeddedRungeKuttaMethod
from leap.vm.expression import collapse_constants
from pymbolic import var
from pymbolic.primitives import CallWithKwargs
from leap.vm.language import (CodeBuilder, SimpleCodeBuilder,
                              TimeIntegratorCode)


class KennedyCarpenterIMEXRungeKuttaBase(EmbeddedRungeKuttaMethod):
    """
    Christopher A. Kennedy, Mark H. Carpenter. Additive Runge-Kutta
    schemes for convection-diffusion-reaction equations.
    Applied Numerical Mathematics
    Volume 44, Issues 1-2, January 2003, Pages 139-181
    http://dx.doi.org/10.1016/S0168-9274(02)00138-1
    """

    def _emit_initialization(self, cbuild):
        """Add code to drive the initialization. Return the list of instruction
        ids."""

        initialization_deps = self._update_rhs_values(cbuild, depends_on=[])
        cbuild.commit()

        return initialization_deps

    def _eval_rhs(self, rhs, y, t):
        return CallWithKwargs(rhs, parameters=(),
                              kw_parameters={"t": t, self._component_id: y})

    def _update_rhs_values(self, cbuild, depends_on):
        """Add code to assign to the explicit and implicit RHSs. Return the
        list of instruction ids."""

        with SimpleCodeBuilder(cbuild, depends_on) as builder:
            builder.assign(self._rhs_expl, self._eval_rhs(
                    self._rhs_expl_function, self._state, self._t))
            builder.assign(self._rhs_impl, self._eval_rhs(
                    self._rhs_impl_function, self._state, self._t))

        return builder.last_added_instruction_id

    def _finish_solution(self, builder, dest, coeffs, explicit_rhss,
                         implicit_rhss):
        """Add code to build the final component in the primary stage. Return
        the list of instruction ids.

        :arg cbuild: The CodeBuilder

        :arg dest: The variable to assign the state to

        :arg coeffs: The coefficients for the linear combination

        :arg explicit_rhss: The list of explicit RHS stage values

        :arg implicit_rhss: The list of implicit RHS stage values
        """

        assert (len(coeffs) == len(explicit_rhss) == len(implicit_rhss))
        args = [(1, self._state)] + [(self._dt * coeff, erhs)
                                     for coeff, erhs in zip(coeffs + coeffs,
                                         explicit_rhss + implicit_rhss)
                                     if coeff]

        builder.assign(dest, sum(arg[0] * arg[1] for arg in args))
        return builder.last_added_instruction_id

    def _emit_primary(self, cbuild):
        """Add code to drive the primary stage. Return the list of instruction
        ids."""

        explicit_rhss = []
        implicit_rhss = []

        with SimpleCodeBuilder(cbuild) as builder:

            # Stage loop
            for c, coeffs_expl, coeffs_impl in \
                    zip(self.c, self.a_explicit, self.a_implicit):

                if len(coeffs_expl) == 0:
                    assert c == 0
                    assert len(coeffs_impl) == 0
                    this_rhs_expl = self._rhs_expl
                    this_rhs_impl = self._rhs_impl
                else:
                    assert len(coeffs_expl) == len(explicit_rhss)
                    assert len(coeffs_impl) == len(implicit_rhss)

                    args = [(1, self._state)] + [
                        (self._dt * coeff, erhs)
                        for coeff, erhs in zip(
                            coeffs_expl + coeffs_impl,
                            explicit_rhss + implicit_rhss)
                        if coeff]

                    sub_y = sum(arg[0] * arg[1] for arg in args)

                    # Compute the next implicit right hand side.
                    # TODO: Compute a guess parameter using stage value
                    # predictors.
                    this_rhs_impl = var(cbuild.fresh_var_name('rhs_impl'))
                    solve_component = var('rhs_impl*')
                    solve_expression = collapse_constants(
                        solve_component - self._eval_rhs(
                            self._rhs_impl_function, sub_y + self.gamma *
                            self._dt * solve_component,
                            self._t + c * self._dt),
                        [solve_component], builder.assign, builder.fresh_var)
                    builder.assign_solved(this_rhs_impl, solve_component,
                                          solve_expression, implicit_rhss[-1],
                                          self._solver_id)

                    # Compute the next explicit right hand side.
                    this_rhs_expl = var(cbuild.fresh_var_name('rhs_expl'))
                    builder.assign(this_rhs_expl, self._eval_rhs(
                            self._rhs_expl_function, sub_y + self.gamma *
                            self._dt * this_rhs_impl,
                            self._t + c * self._dt))

                explicit_rhss.append(this_rhs_expl)
                implicit_rhss.append(this_rhs_impl)

            if not self.adaptive:
                state_update_ids = self._finish_solution(builder,
                    self._state, self.high_order_coeffs if
                    self.use_high_order else self.low_order_coeffs,
                    explicit_rhss, implicit_rhss)
            else:
                high_order_end_state = \
                    var(cbuild.fresh_var_name('high_order_end_state'))
                low_order_end_state = \
                    var(cbuild.fresh_var_name('low_order_end_state'))

                adaptive_ids = set(self._finish_solution(builder,
                    high_order_end_state, self.high_order_coeffs,
                    explicit_rhss, implicit_rhss))

                adaptive_ids |= set(self._finish_solution(builder,
                    low_order_end_state, self.low_order_coeffs,
                    explicit_rhss, implicit_rhss))

        # Done with the part that uses SimpleCodeBuilder. Finish the
        # work by either incrementing t and returning the state or by
        # adapting the step size.

        if not self.adaptive:
            last_work_ids = self.add_ret_state_and_increment_t(
                cbuild, self._state, self._component_id,
                self._t + self._dt, state_update_ids)
        else:
            if self.limiter_name is not None:
                limiter = var("<func>" + self.limiter_name)
            else:
                limiter = lambda x: x
            last_work_ids = self.adapt_step_size(cbuild, self._state,
                                                 self._component_id,
                                                 self._t, self._dt,
                                                 high_order_end_state,
                                                 low_order_end_state,
                                                 adaptive_ids, limiter)
            cbuild.commit()

        return self._update_rhs_values(cbuild, last_work_ids)

    def __call__(self, component_id, solver_id):
        self._dt = var('<dt>')
        self._t = var('<t>')
        self._state = var('<state>' + component_id)
        self._component_id = component_id
        self._solver_id = solver_id

        # Saved implicit and explicit right hand sides
        self._rhs_expl = var('<p>rhs_expl')
        self._rhs_impl = var('<p>rhs_impl')

        # Implicit and explicit right hand side functions
        self._rhs_expl_function = var('<func>expl_' + self._component_id)
        self._rhs_impl_function = var('<func>impl_' + self._component_id)

        cbuild = CodeBuilder()

        initialization_deps = self._emit_initialization(cbuild)
        primary_deps = self._emit_primary(cbuild)

        return TimeIntegratorCode(
            instructions=cbuild.instructions,
            initialization_dep_on=initialization_deps,
            step_dep_on=primary_deps,
            step_before_fail=True)


class KennedyCarpenterIMEXARK4(KennedyCarpenterIMEXRungeKuttaBase):
    gamma = 1./4.

    c = [0, 1/2, 83/250, 31/50, 17/20, 1]
    low_order = 3
    high_order = 4

    high_order_coeffs = [
            82889/524892,
            0,
            15625/83664,
            69875/102672,
            -2260/8211,
            1/4]

    low_order_coeffs = [
            4586570599/29645900160,
            0,
            178811875/945068544,
            814220225/1159782912,
            -3700637/11593932,
            61727/225920
            ]

    # ARK4(3)6L[2]SA-ERK (explicit)
    a_explicit = [[],
            [1/2],
            [13861/62500, 6889/62500],
            [-116923316275/2393684061468,
                -2731218467317/15368042101831,
                9408046702089/11113171139209],
            [-451086348788/2902428689909,
                -2682348792572/7519795681897,
                12662868775082/11960479115383,
                3355817975965/11060851509271],
            [647845179188/3216320057751,
                73281519250/8382639484533,
                552539513391/3454668386233,
                3354512671639/8306763924573,
                4040/17871]]

    # ARK4(3)6L[2]SA-ESDIRK (implicit)
    a_implicit = [[],
            [1/4],
            [8611/62500, -1743/31250],
            [5012029/34652500, -654441/2922500, 174375/388108],
            [15267082809/155376265600, -71443401/120774400,
                730878875/902184768, 2285395/8070912],
            [82889/524892, 0, 15625/83664, 69875/102672,
                -2260/8211]]

    assert (len(a_explicit) == len(a_implicit)
            == len(low_order_coeffs)
            == len(high_order_coeffs)
            == len(c))
