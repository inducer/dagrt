"""Runge-Kutta ODE timestepper."""

from __future__ import division

__copyright__ = "Copyright (C) 2007-2013 Andreas Kloeckner"

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

class MultirateFastestFirstEulerMethod(Method):

    # Add code for y' = y + dt * f(t, y).

    def euler_update(self, cbuild, funcs, deps, t, dt, fast, slow,
                     lhs, suffix):
        add_and_get_ids = cbuild.add_and_get_ids

        from leap.vm.language import AssignRHS, AssignNorm, \
            AssignExpression, ReturnState, If, Raise, FailStep, \
            TimeIntegratorCode, CodeBuilder

        from pymbolic import var

        rhss = []
        
        # Compute all derivatives.
        for f in funcs:
            rhs = 'rhs' + str(len(rhss)) + '_' + suffix
            add_and_get_ids(AssignRHS(
                assignees=(rhs, ),
                component_id=f,
                t=t,
                rhs_arguments=((('u', fast), ('v', slow)), ),
                id='assign_' + rhs,
                depends_on=deps,
                ))
            rhss.append(rhs)

        # Take sum of derivatives * dt
        add_and_get_ids(AssignExpression(lhs.name, lhs + dt
                        * sum(map(var, rhss)), id='update_state_'
                        + suffix, depends_on=map(lambda x: 'assign_' \
                        + x, rhss)))

        return 'update_state_' + suffix

    def __call__(self, components):
        from leap.vm.language import AssignRHS, AssignNorm, \
            AssignExpression, ReturnState, If, Raise, FailStep, \
            TimeIntegratorCode, CodeBuilder

        cbuild = CodeBuilder()

        add_and_get_ids = cbuild.add_and_get_ids

        from pymbolic import var

        # Ratio of slow to fast (must be integer)
        factor = var('<state>factor')

        dt = var('<dt>')
        t = var('<t>')
        # Slow component
        slow = var('<p>slow')
        # Computed derivative of slow component at last synchronization
        slowrate = var('<p>slow_rate')
        # Fast component
        fast = var('<p>fast')
        # Value of fast component at last slow component synchronization
        fastprev = var('<p>fast_prev')
        # Distance in timesteps since last component synchronization
        step = var('<p>step')
        # Extrapolated value of slow component
        slow_extrapolate = var('slow_extrapolate')

        dep_inf_exclude_names = [v.name for v in [
            dt, t, slow, slowrate, fast, fastprev, step, slow_extrapolate
            ]]

        # Initialization

        initialization_dep_on = add_and_get_ids(
            AssignExpression(step.name, 0, id='init_step'),
            AssignExpression(fast.name, var('<state>fast'),
                             id='init_fast'),
            AssignExpression(slow.name, var('<state>slow'),
                             id='init_slow'),
            AssignRHS(
                assignees=('slow_rate_init_s2s', ),
                component_id='s2s',
                t=t,
                rhs_arguments=((('u', fast), ('v', slow)), ),
                id='init_s2s',
                depends_on=['init_slow', 'init_fast'],
                ),
            AssignRHS(
                assignees=('slow_rate_init_f2s', ),
                component_id='f2s',
                t=t,
                rhs_arguments=((('u', fast), ('v', slow)), ),
                id='init_f2s',
                depends_on=['init_slow', 'init_fast'],
                ),
            AssignExpression(slowrate.name, var('slow_rate_init_f2s')
                             + var('slow_rate_init_s2s'),
                             depends_on=['init_s2s', 'init_f2s']),
            AssignExpression(fastprev.name, fast,
                             depends_on=['init_fast']),
            )

        cbuild.commit()

        from pymbolic.primitives import Comparison

        # Extrapolation

        add_and_get_ids(AssignExpression(slow_extrapolate.name, slow
                        + step * dt * slowrate, id='extrapolation'))

        # Fast step with extrapolated value

        fast_step = self.euler_update(
            cbuild, ['f2f', 's2f'], ['extrapolation'], t, dt, fast,
            slow_extrapolate, fast, 'fast'
            )

        # Slow step

        slow_step = self.euler_update(
            cbuild, ['f2s', 's2s'], [], t - (factor - 1) * dt, factor * dt,
            fastprev, slow, slow, 'slow'
            )

        # Upon completion of the slow step, the new derivative needs to be computed for
        # extrapolation purposes.

        add_and_get_ids(AssignRHS(assignees=('u_rhs0_slow', ),
                        component_id='f2s', t=t + dt,
                        rhs_arguments=((('u', fast), ('v', slow)), ),
                        depends_on=[slow_step, fast_step]),

                        AssignRHS(assignees=('u_rhs1_slow', ),
                        component_id='s2s', t=t + dt,
                        rhs_arguments=((('u', fast), ('v', slow)), ),
                        depends_on=[slow_step, fast_step]),

                        AssignExpression(slowrate.name,
                        var('u_rhs0_slow') + var('u_rhs1_slow'),
                        id='update_slowrate'))

        # Upon completion of the slow step, the value of fast at that time must be
        # recorded.

        (update_fastprev, ) = \
            add_and_get_ids(AssignExpression(fastprev.name, fast,
                            depends_on=[fast_step, slow_step]))

        # This condition determines if the slow step is taken.

        add_and_get_ids(If(condition=Comparison(step, '==', factor
                        - 1), then_depends_on=[fast_step, slow_step,
                        'update_slowrate', update_fastprev],
                        else_depends_on=[fast_step], id='decide_step'))

        # Step depends on: (1) return value (fast component), (2) incrementing <t>,
        # (3) incrementing <step>

        add_and_get_ids(ReturnState(
            id='ret',
            time_id='final',
            time=t + dt,
            component_id='<state>',
            expression=fast,
            depends_on=['decide_step'],
            ),
            AssignExpression('<t>', t + dt, id='increment_t',
                             depends_on=['ret']),
            AssignExpression(step.name, (step + 1) % factor,
                             depends_on=['decide_step'],
                             id='update_step'))

        cbuild.infer_single_writer_dependencies(exclude=dep_inf_exclude_names)
        cbuild.commit()

        return TimeIntegratorCode(instructions=cbuild.instructions,
                                  initialization_dep_on=initialization_dep_on,
                                  step_dep_on=['ret', 'increment_t',
                                  'update_step'],
                                  step_before_fail=False)

# {{{ Embedded Runge-Kutta schemes base class

class EmbeddedRungeKuttaMethod(Method):
    def __init__(self, use_high_order=True,
            atol=0, rtol=0, max_dt_growth=5, min_dt_shrinkage=0.1,
            limiter_name=None):
        self.limiter_name = limiter_name

        self.use_high_order = use_high_order

        self.adaptive = bool(atol or rtol)
        self.atol = atol
        self.rtol = rtol

        self.max_dt_growth = max_dt_growth
        self.min_dt_shrinkage = min_dt_shrinkage


class EmbeddedButcherTableauMethod(EmbeddedRungeKuttaMethod):
    def __call__(self, component_id):
        """
        :arg component_id: an identifier to be used for the single state component
            supported.
        """
        from leap.vm.language import (
                AssignRHS, AssignNorm, AssignExpression,
                ReturnState, If, Raise, FailStep,
                TimeIntegratorCode,
                CodeBuilder)

        cbuild = CodeBuilder()

        add_and_get_ids = cbuild.add_and_get_ids

        from pymbolic import var

        dt = var("<dt>")
        t = var("<t>")
        last_rhs = var("<p>last_rhs_"+component_id)
        state = var("<state>"+component_id)

        dep_inf_exclude_names = ["<t>", "<dt>", state.name, last_rhs.name]

        if self.limiter_name is not None:
            limiter = var("<func>"+self.limiter_name)
        else:
            limiter = lambda x: x

        initialization_dep_on = add_and_get_ids(
                AssignRHS(
                    assignees=(last_rhs.name,),
                    component_id=component_id,
                    t=var("<t>"),
                    rhs_arguments=(((component_id, state),),)))

        cbuild.commit()

        rhss = []

        all_rhs_eval_ids = []

        # {{{ stage loop

        for istage, (c, coeffs) in enumerate(self.butcher_tableau):
            if len(coeffs) == 0:
                assert c == 0
                this_rhs = last_rhs
            else:
                stage_state = limiter(
                        state + sum(
                            dt * coeff * rhss[j]
                            for j, coeff in enumerate(coeffs)))

                rhs_id = "rhs%d" % istage
                rhs_insn_id = "ev_rhs%d" % istage
                all_rhs_eval_ids.append(rhs_insn_id)

                add_and_get_ids(
                        AssignRHS(
                            assignees=(rhs_id,),
                            component_id=component_id,
                            t=t + c*dt,
                            rhs_arguments=(((component_id, stage_state),),),
                            id=rhs_insn_id,
                            ))

                this_rhs = var(rhs_id)

            rhss.append(this_rhs)

        # }}}

        last_rhs_assignment_id, = add_and_get_ids(
                AssignExpression(last_rhs.name, this_rhs))

        if not self.adaptive:
            if self.use_high_order:
                coeffs = self.high_order_coeffs
            else:
                coeffs = self.low_order_coeffs

            add_and_get_ids(
                    AssignExpression(
                        state.name, limiter(
                            state + sum(
                                dt * coeff * rhss[j]
                                for j, coeff in enumerate(coeffs))),
                        id="update_state",
                        # don't change state before all RHSs are done using it
                        depends_on=all_rhs_eval_ids),

                    ReturnState(
                        id="ret_state",
                        time_id="final",
                        time=t + dt,
                        component_id=component_id,
                        expression=state,
                        depends_on=["update_state"]),

                    AssignExpression(
                        "<t>", t + dt,
                        id="increment_t",
                        depends_on=["update_state", "ret_state"],
                        ),
                    )

            cbuild.infer_single_writer_dependencies(exclude=dep_inf_exclude_names)
            cbuild.commit()

            return TimeIntegratorCode(
                    instructions=cbuild.instructions,
                    initialization_dep_on=initialization_dep_on,
                    step_dep_on=[
                        "ret_state",
                        last_rhs_assignment_id,
                        "increment_t"],
                    step_before_fail=False)

        else:
            # {{{ step size adaptation

            from leap.method import TimeStepUnderflow

            from pymbolic.primitives import Min, Max, Comparison, LogicalOr
            add_and_get_ids(
                    AssignExpression(
                        "high_order_end_state", state + sum(
                                dt * coeff * rhss[j]
                                for j, coeff in enumerate(self.high_order_coeffs)),
                        id="compute_hes"),
                    AssignExpression(
                        "low_order_end_state", state + sum(
                                dt * coeff * rhss[j]
                                for j, coeff in enumerate(self.low_order_coeffs)),
                        id="compute_les"),
                    AssignNorm("norm_start_state", state,
                        id="compute_state_norm"),
                    AssignNorm("norm_end_state", var("low_order_end_state")),
                    AssignNorm(
                        "rel_error_raw", (
                            var("high_order_end_state") - var("low_order_end_state")
                            ) / (
                                var("len")(state)**0.5
                                *
                                (self.atol + self.rtol * Max((
                                    var("norm_start_state"),
                                    var("norm_end_state"))))
                                )
                            ),
                    If(
                        condition=Comparison(var("rel_error_raw"), "==", 0),
                        then_depends_on=["rel_err_zero"],
                        else_depends_on=["rel_err_nonzero"],
                        id="rel_error_zero_check"),
                    # then
                    AssignExpression("rel_error", var("rel_error_raw"),
                        id="rel_err_nonzero"),
                    # else
                    AssignExpression("rel_error", 1e-14,
                        id="rel_err_zero"),
                    # endif

                    If(
                        condition=LogicalOr((
                            Comparison(var("rel_error"), ">", 1),
                            var("isnan")(var("rel_error"))
                            )),
                        then_depends_on=["rej_step"],
                        else_depends_on=["acc_adjust_dt", last_rhs_assignment_id,
                            "ret_state", "increment_t"],
                        depends_on=["rel_error_zero_check"],
                        id="reject_check"),
                    # then
                        # reject step

                        If(
                            condition=var("isnan")(var("rel_error")),
                            then_depends_on=["min_adjust_dt"],
                            else_depends_on=["rej_adjust_dt"],
                            depends_on=["rel_error_zero_check"],
                            id="adjust_dt"),
                        # then
                        AssignExpression("<dt>",
                            self.min_dt_shrinkage*dt,
                            id="min_adjust_dt"),
                        # else
                        AssignExpression("<dt>",
                            Max((
                                0.9 * dt * var("rel_error")**(-1/self.low_order),
                                self.min_dt_shrinkage * dt)),
                            id="rej_adjust_dt"),
                        # endif

                        If(
                            condition=Comparison(t + dt, "==", t),
                            then_depends_on=["tstep_underflow"],
                            else_depends_on=[],
                            id="check_underflow",
                            depends_on=["adjust_dt"]),
                        # then
                        Raise(TimeStepUnderflow, id="tstep_underflow"),
                        # endif

                        FailStep(
                            id="rej_step",
                            depends_on=["check_underflow"]),

                    # else
                        # accept step

                        AssignExpression("<dt>",
                            Min((
                                0.9 * dt * var("rel_error")**(-1/self.high_order),
                                self.max_dt_growth * dt)),
                            id="acc_adjust_dt",
                            depends_on=["increment_t"],
                            ),
                        AssignExpression(
                            state.name, limiter(var("high_order_end_state")),
                            id="update_state",
                            depends_on=["compute_les", "compute_hes",
                                "compute_state_norm"]),

                        ReturnState(
                            id="ret_state",
                            time_id="final",
                            time=t + dt,
                            component_id=component_id,
                            expression=state,
                            depends_on=["update_state"]),

                        AssignExpression(
                            "<t>", t + dt,
                            id="increment_t",
                            depends_on=["ret_state"]),
                    # endif
                    )

            cbuild.infer_single_writer_dependencies(exclude=dep_inf_exclude_names)
            cbuild.commit()

            return TimeIntegratorCode(
                    instructions=cbuild.instructions,
                    initialization_dep_on=initialization_dep_on,
                    step_dep_on=["reject_check"],
                    step_before_fail=False)

            # }}}

# }}}


# {{{ Bogacki-Shampine second/third-order Runge-Kutta

class ODE23TimeStepper(EmbeddedButcherTableauMethod):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    http://dx.doi.org/10.1016/0893-9659(89)90079-7
    """

    dt_fudge_factor = 1

    butcher_tableau = [
            (0, []),
            (1/2, [1/2]),
            (3/4, [0, 3/4]),
            (1, [2/9, 1/3, 4/9])
            ]

    low_order = 2
    low_order_coeffs = [7/24, 1/4, 1/3, 1/8]
    high_order = 3
    high_order_coeffs = [2/9, 1/3, 4/9, 0]

# }}}


# {{{ Dormand-Prince fourth/fifth-order Runge-Kutta

class ODE45TimeStepper(EmbeddedButcherTableauMethod):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    http://dx.doi.org/10.1016/0771-050X(80)90013-3.
    """

    dt_fudge_factor = 1

    butcher_tableau = [
            (0, []),
            (1/5, [1/5]),
            (3/10, [3/40, 9/40]),
            (4/5, [44/45, -56/15, 32/9]),
            (8/9, [19372/6561, -25360/2187, 64448/6561, -212/729]),
            (1, [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
            (1, [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
            ]

    low_order = 4
    low_order_coeffs = [5179/57600, 0, 7571/16695, 393/640, -92097/339200,
            187/2100, 1/40]
    high_order = 5
    high_order_coeffs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# }}}


# vim: foldmethod=marker
