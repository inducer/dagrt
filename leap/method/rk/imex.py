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


class KennedyCarpenterIMEXRungeKuttaBase(EmbeddedRungeKuttaMethod):
    """
    Christopher A. Kennedy, Mark H. Carpenter. Additive Runge-Kutta
    schemes for convection-diffusion-reaction equations.
    Applied Numerical Mathematics
    Volume 44, Issues 1-2, January 2003, Pages 139-181
    http://dx.doi.org/10.1016/S0168-9274(02)00138-1

    Context:
        state: The value that is integrated
        rhs_expl_func: The explicit right hand side function
        rhs_impl_func: The implicit right hand side function
    """

    def __init__(self, component_id, **kwargs):
        EmbeddedRungeKuttaMethod.__init__(self, **kwargs)

        self.dt = var('<dt>')
        self.t = var('<t>')
        self.state = var('<state>' + component_id)
        self.component_id = component_id

        # Saved implicit and explicit right hand sides
        self.rhs_expl = var('<p>rhs_expl')
        self.rhs_impl = var('<p>rhs_impl')

        if self.limiter_name is not None:
            limiter = var("<func>" + self.limiter_name)
        else:
            limiter = lambda x: x
        self.limiter_func = limiter

        # Implicit and explicit right hand side functions
        self.rhs_expl_func = var('<func>expl_' + self.component_id)
        self.rhs_impl_func = var('<func>impl_' + self.component_id)

    def call_rhs(self, t, y, rhs):
        return CallWithKwargs(rhs, parameters=(),
                              kw_parameters={"t": t, self.component_id: y})

    def finish_solution(self, cb, high_order_estimate, high_order_implicit_rhs,
                        low_order_estimate):
        cb.fence()
        next_state = high_order_estimate if self.use_high_order \
                                         else low_order_estimate
        cb(self.state, next_state)
        cb(self.rhs_expl, self.call_rhs(self.t + self.dt, self.state,
                                        self.rhs_expl_func))
        if self.use_high_order:
            cb(self.rhs_impl, high_order_implicit_rhs)
        else:
            cb(self.rhs_impl, self.call_rhs(self.t + self.dt, self.state,
                                            self.rhs_impl_func))
        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')
        cb.fence()
        cb(self.t, self.t + self.dt)

    def emit_primary(self, cb):
        """Add code to drive the primary state."""

        explicit_rhss = []
        implicit_rhss = []

        # Stage loop
        for c, coeffs_expl, coeffs_impl in \
                zip(self.c, self.a_explicit, self.a_implicit):

            if len(coeffs_expl) == 0:
                assert c == 0
                assert len(coeffs_impl) == 0
                this_rhs_expl = self.rhs_expl
                this_rhs_impl = self.rhs_impl
            else:
                assert len(coeffs_expl) == len(explicit_rhss)
                assert len(coeffs_impl) == len(implicit_rhss)

                sub_y_args = [(self.dt * coeff, erhs)
                    for coeff, erhs in zip(
                        coeffs_expl + coeffs_impl,
                        explicit_rhss + implicit_rhss)
                    if coeff]

                sub_y = sum(arg[0] * arg[1] for arg in sub_y_args)

                # Compute the next implicit right hand side for the stage value.
                # TODO: Compute a guess parameter using stage value predictors.
                this_rhs_impl = cb.fresh_var('rhs_impl')
                solve_component = cb.fresh_var('rhs_impl*')
                solve_expression = collapse_constants(
                    solve_component - self.call_rhs(
                        self.t + c * self.dt,
                        self.state + sub_y + self.gamma * self.dt * solve_component,
                        self.rhs_impl_func),
                    [solve_component, self.state], cb.assign, cb.fresh_var)
                cb.assign_solved(this_rhs_impl, solve_component,
                                 solve_expression, implicit_rhss[-1], 0)

                # Compute the next explicit right hand side for the stage value.
                this_rhs_expl = cb.fresh_var('rhs_expl')
                cb.assign(this_rhs_expl, self.call_rhs(
                          self.t + c * self.dt,
                          self.state + sub_y + self.gamma * self.dt * this_rhs_impl,
                          self.rhs_expl_func))

            explicit_rhss.append(this_rhs_expl)
            implicit_rhss.append(this_rhs_impl)

        # Compute solution estimates.
        high_order_estimate = cb.fresh_var('high_order_estimate')
        low_order_estimate = cb.fresh_var('low_order_estimate')

        def combine(coeffs):
            assert (len(coeffs) == len(explicit_rhss) == len(implicit_rhss))
            args = [(1, self.state)] + [
                    (self.dt * coeff, erhs)
                    for coeff, erhs in zip(coeffs + coeffs,
                        explicit_rhss + implicit_rhss)
                    if coeff]
            return sum(arg[0] * arg[1] for arg in args)

        cb(high_order_estimate, combine(self.high_order_coeffs))
        cb(low_order_estimate, combine(self.low_order_coeffs))

        high_order_implicit_rhs = implicit_rhss[-1]

        if not self.adaptive:
            self.finish_solution(cb, high_order_estimate,
                                 high_order_implicit_rhs, low_order_estimate)
        else:
            self.finish_adaptive(cb, high_order_estimate,
                                 high_order_implicit_rhs, low_order_estimate)

    def implicit_expression(self, expression_tag=None):
        from leap.vm.expression import parse
        return (parse("`solve_component` - `{rhs_impl}`(t=t, {component_id}="
                      "`{state}` + sub_y + coeff * `solve_component`)".format(
                          component_id=self.component_id,
                          rhs_impl=self.rhs_impl_func.name,
                          state=self.state.name)), "solve_component")

    def generate(self, solver_hook):
        from leap.vm.language import NewCodeBuilder, TimeIntegratorCode

        with NewCodeBuilder(label="initialization") as cb_init:
            cb_init.assign(self.rhs_expl, self.call_rhs(
                    self.t, self.state, self.rhs_expl_func))
            cb_init.assign(self.rhs_impl, self.call_rhs(
                    self.t, self.state, self.rhs_impl_func))

        with NewCodeBuilder(label="primary") as cb_primary:
            self.emit_primary(cb_primary)

        from leap.vm.implicit import replace_AssignSolved
        code = TimeIntegratorCode.create_with_init_and_step(
            instructions=cb_init.instructions | cb_primary.instructions,
            initialization_dep_on=cb_init.state_dependencies,
            step_dep_on=cb_primary.state_dependencies)

        return replace_AssignSolved(code, solver_hook)


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
