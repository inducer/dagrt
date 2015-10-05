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


from pymbolic import var
from leap.method import TwoOrderAdaptiveMethod
from leap.method.rk import ButcherTableauMethod


class KennedyCarpenterIMEXRungeKuttaMethodBase(
        TwoOrderAdaptiveMethod, ButcherTableauMethod):
    """
    Christopher A. Kennedy, Mark H. Carpenter. Additive Runge-Kutta
    schemes for convection-diffusion-reaction equations.
    Applied Numerical Mathematics
    Volume 44, Issues 1-2, January 2003, Pages 139-181
    http://dx.doi.org/10.1016/S0168-9274(02)00138-1

    User-supplied context:
        <state> + component_id: The value that is integrated
        <func>rhs_expl_ + component_id: The explicit right hand side function
        <func>rhs_impl_ + component_id: The implicit right hand side function
    """

    def __init__(self, component_id, use_high_order=True, limiter_name=None,
            use_explicit=True, use_implicit=True,
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
        self.use_explicit = use_explicit
        self.use_implicit = use_implicit

    def implicit_expression(self, expression_tag=None):
        from leap.vm.expression import parse
        return (parse("`solve_component` - `<func>impl_{component_id}`("
            "t=t, {component_id}="
            "`{state}` + sub_y + coeff * `solve_component`)".format(
                component_id=self.component_id,
                state=self.state.name)), "solve_component")

    def generate(self):
        if self.use_high_order:
            estimate_names = ("high_order", "low_order")
        else:
            estimate_names = ("low_order", "high_order")

        set_names = []
        if self.use_implicit:
            set_names.append("implicit")
        if self.use_explicit:
            set_names.append("explicit")

        return self.generate_butcher(
                stage_coeff_set_names=set_names,
                stage_coeff_sets={
                    "explicit": self.a_explicit,
                    "implicit": self.a_implicit,
                    },
                rhs_funcs={
                    "implicit": var("<func>impl_"+self.component_id),
                    "explicit": var("<func>expl_"+self.component_id),
                    },
                estimate_coeff_set_names=estimate_names,
                estimate_coeff_sets={
                    "high_order": self.high_order_coeffs,
                    "low_order": self.low_order_coeffs
                    })

    def finish(self, cb, estimate_coeff_set_names, estimate_coeff_sets):
        if not self.adaptive:
            super(KennedyCarpenterIMEXRungeKuttaMethodBase, self).finish(
                    cb, estimate_coeff_set_names, estimate_coeff_sets)
        else:
            high_est = estimate_coeff_sets[
                    estimate_coeff_set_names.index("high_order")]
            low_est = estimate_coeff_sets[
                    estimate_coeff_set_names.index("low_order")]
            self.finish_adaptive(cb, high_est, low_est)

    def finish_nonadaptive(self, cb, high_order_estimate, low_order_estimate):
        if self.use_high_order:
            est = high_order_estimate
        else:
            est = low_order_estimate

        cb(self.state, est)
        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')


class KennedyCarpenterIMEXARK4Method(KennedyCarpenterIMEXRungeKuttaMethodBase):
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
            [1/4, gamma],
            [8611/62500, -1743/31250, gamma],
            [5012029/34652500, -654441/2922500, 174375/388108, gamma],
            [15267082809/155376265600, -71443401/120774400,
                730878875/902184768, 2285395/8070912, gamma],
            [82889/524892, 0, 15625/83664, 69875/102672,
                -2260/8211, gamma]]

    recycle_last_stage_coeff_set_names = ("implicit",)

    assert (len(a_explicit) == len(a_implicit)
            == len(low_order_coeffs)
            == len(high_order_coeffs)
            == len(c))
