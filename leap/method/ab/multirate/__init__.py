# -*- coding: utf-8 -*-

"""Multirate-AB ODE solver."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
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

import numpy
from pytools import memoize_method
from leap.method.ab import AdamsBashforthTimeStepperBase
from leap.method.ab.utils import make_generic_ab_coefficients, linear_comb
from leap.method.ab.multirate.methods import (HIST_NAMES, HIST_F2F, HIST_S2F,
                                              HIST_F2S, HIST_S2S)
from leap.method.ab.multirate.processors import MRABProcessor
from pymbolic import var


__doc__ = """
.. autoclass:: TwoRateAdamsBashforthTimeStepper
"""


class TwoRateAdamsBashforthTimeStepper(AdamsBashforthTimeStepperBase):
    """Simultaneously timesteps two parts of an ODE system,
    the first with a small timestep, the second with a large timestep.

    [1] C.W. Gear and D.R. Wells, "Multirate linear multistep methods," BIT
    Numerical Mathematics,  vol. 24, Dec. 1984,pg. 484-502.

    User-supplied context:
        <state>slow: The slow value to be integrated
        <state>fast: The fast value to be integrated
        <func>f2f: The fast-to-fast coupling
        <func>s2f: The slow-to-fast coupling
        <func>s2s: The slow-to-slow coupling
        <func>f2s: The fast-to-slow coupling
    """

    def __init__(self, method, orders, substep_count):
        super(TwoRateAdamsBashforthTimeStepper, self).__init__()
        self.method = method

        # Variables
        from pymbolic import var

        self.t = var('<t>')
        self.dt = var('<dt>')
        self.step = var('<p>step')

        # Slow and fast components
        self.slow = var('<state>slow')
        self.fast = var('<state>fast')

        # Individual component functions
        self.f2f = var('<func>f2f')
        self.s2f = var('<func>s2f')
        self.s2s = var('<func>s2s')
        self.f2s = var('<func>f2s')

        # Current values of components
        self.current_rhss = {
            HIST_F2F: var('<p>f2f_n'),
            HIST_S2F: var('<p>s2f_n'),
            HIST_F2S: var('<p>f2s_n'),
            HIST_S2S: var('<p>s2s_n')
            }

        self.component_functions = {
            HIST_F2S: var('<func>f2s'),
            HIST_F2F: var('<func>f2f'),
            HIST_S2S: var('<func>s2s'),
            HIST_S2F: var('<func>s2f')
            }

        self.large_dt = self.dt
        self.small_dt = self.dt / substep_count
        self.substep_count = substep_count

        self.orders = {
                HIST_F2F: orders['f2f'],
                HIST_S2F: orders['s2f'],
                HIST_F2S: orders['f2s'],
                HIST_S2S: orders['s2s'],
                }

        self.max_order = max(self.orders.values())

        self.histories = {}

        for component in HIST_NAMES:
            name = component().__class__.__name__.lower()
            var_names = [self.current_rhss[component]]
            for past in range(1, self.orders[component]):
                var_names.append(var('<p>' + name + '_n_minus_' + str(past)))
            self.histories[component] = var_names

        self.time_histories = {}

        for component in HIST_NAMES:
            name = component().__class__.__name__.lower()
            time_var_names = [var('time' + name + '_n')]
            for past in range(1, self.orders[component]):
                time_var_names.append(var('time' + name + '_n_minus_' + str(past)))
            self.time_histories[component] = time_var_names

        self.hist_is_fast = {
                HIST_F2F: True,
                HIST_S2F: self.method.s2f_hist_is_fast,
                HIST_S2S: False,
                HIST_F2S: False
                }

    @memoize_method
    def get_coefficients(self, for_fast_history, hist_head_time_level,
                         start_level, end_level, order):

        history_times = numpy.arange(0, -order, -1, dtype=numpy.float64)

        if for_fast_history:
            history_times /= self.substep_count

        history_times += hist_head_time_level/self.substep_count

        t_start = start_level / self.substep_count
        t_end = end_level / self.substep_count

        return make_generic_ab_coefficients(history_times, t_start, t_end)

    def compute_history_assignments(self):
        """
        Compute how history values should be assigned during RK initialization.

        Return a list `assign_before`, where each `assign_before[i]` maps
        variable names to RHS components. If `var` is in `assign_before[i]`,
        then before initialization step `i` is executed, `var` should be
        assigned the value of the RHS component `assign_before[i][var]`.
        """

        initialization_steps = self.max_order - 1
        total_substeps = initialization_steps * self.substep_count

        assign_before = [{} for step in range(total_substeps)]
        for component in HIST_NAMES:
            history = list(range(total_substeps + 1))
            history.reverse()
            if not self.hist_is_fast[component]:
                history = history[::self.substep_count]
            history = history[:self.orders[component]]
            for index, entry in enumerate(history):
                if index == 0:
                    # We don't store the most recent entry, this is already
                    # assumed to be initialized.
                    continue
                variable = self.histories[component][index]
                assign_before[entry][variable] = component
            assert len(self.histories[component]) == self.orders[component]

        return assign_before

    def emit_initialization(self, cb):
        """Initialize method variables."""
        cb(self.step, 0)
        # Initial value of RK derivatives
        for hist_component, function in self.component_functions.items():
            assignee = self.current_rhss[hist_component]
            cb(assignee, function(t=self.t, s=self.slow, f=self.fast))

    def emit_small_rk_step(self, cb, t, name_prefix):
        """Emit a single step of an RK method."""

        rk_tableau, rk_coeffs = self.get_rk_tableau_and_coeffs(self.max_order)

        make_stage_history = lambda prefix: \
            [var(prefix + str(i)) for i in range(len(rk_tableau))]
        stage_rhss = {
            HIST_F2F: make_stage_history(name_prefix + '_rk_f2f_'),
            HIST_S2F: make_stage_history(name_prefix + '_rk_s2f_'),
            HIST_F2S: make_stage_history(name_prefix + '_rk_f2s_'),
            HIST_S2S: make_stage_history(name_prefix + '_rk_s2s_')
            }

        for stage_number, (c, coeffs) in enumerate(rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                for component in HIST_NAMES:
                    cb(stage_rhss[component][stage_number],
                       self.current_rhss[component])
            else:
                stage_s = self.slow + sum(self.small_dt * coeff *
                                          (stage_rhss[HIST_S2S][k] +
                                           stage_rhss[HIST_F2S][k])
                                          for k, coeff in enumerate(coeffs))

                stage_f = self.fast + sum(self.small_dt * coeff *
                                          (stage_rhss[HIST_S2F][k] +
                                           stage_rhss[HIST_F2F][k])
                                          for k, coeff in enumerate(coeffs))

                for component, function in self.component_functions.items():
                    cb(stage_rhss[component][stage_number],
                       function(t=t + c * self.small_dt, s=stage_s, f=stage_f))

        cb.fence()

        cb(self.slow, self.slow + self.small_dt *
                      sum(coeff * (stage_rhss[HIST_F2S][k] +
                                    stage_rhss[HIST_S2S][k])
                      for k, coeff in enumerate(rk_coeffs)))

        cb(self.fast, self.fast + self.small_dt *
                      sum(coeff * (stage_rhss[HIST_F2F][k] +
                                   stage_rhss[HIST_S2F][k])
                      for k, coeff in enumerate(rk_coeffs)))

        for hist_component, function in self.component_functions.items():
            assignee = self.current_rhss[hist_component]
            cb(assignee, function(t=t + self.small_dt, s=self.slow,
                                  f=self.fast))

    def emit_rk_startup(self, cb):
        """Initialize the stepper with an RK method. Return the code that
        computes the startup history."""

        initialization_steps = self.max_order - 1
        assert initialization_steps > 0

        assign_before = self.compute_history_assignments()

        # Try initializing time histories

        for hn in HIST_NAMES:
            time_hist = self.time_histories[hn]
            for i in range(len(time_hist)):
                cb(time_hist[i], 0)
                cb.fence()

        for substep_index in range(self.substep_count):
            # Add any assignments that need to be run ahead of this
            # substep.
            for step in range(initialization_steps):
                substep = step * self.substep_count + substep_index
                if not assign_before[substep]:
                    continue
                with cb.if_(self.step, "==", step):
                    for name, component in assign_before[substep].items():
                        cb(name, self.current_rhss[component])

            # Emit the RK substep body.
            time = self.t + substep_index * self.small_dt
            name_prefix = 'substep' + str(substep_index)
            self.emit_small_rk_step(cb, time, name_prefix)

        # Increment the current step after taking all the substeps.
        cb(self.step, self.step + 1)

        return cb

    def emit_ab_method(self, cb):
        """Add code for the main Adams-Bashforth method."""
        rhss = [self.f2f, self.s2f, self.f2s, self.s2s]
        codegen = MRABCodeEmitter(self, cb, (self.fast, self.slow), self.t, rhss)
        codegen.run()

    def emit_epilogue(self, cb):
        """Add code that finished a timestep."""
        cb.yield_state(self.slow, "slow", self.t + self.dt, "final")
        cb.yield_state(self.fast, "fast", self.t + self.dt, "final")
        cb.fence()
        cb(self.t, self.t + self.dt)

    def generate(self):
        from leap.vm.language import (TimeIntegratorCode, TimeIntegratorState,
                                      CodeBuilder)

        # Initialization state
        with CodeBuilder(label="initialization") as cb_init:
            self.emit_initialization(cb_init)

        # Primary state
        with CodeBuilder(label="primary") as cb_primary:
            self.emit_ab_method(cb_primary)
            self.emit_epilogue(cb_primary)

        bootstrap_steps = self.max_order - 1

        if bootstrap_steps == 0:
            # No need for bootstrapping - just return the primary code.
            return TimeIntegratorCode.create_with_init_and_step(
                instructions=cb_init.instructions | cb_primary.instructions,
                initialization_dep_on=cb_init.state_dependencies,
                step_dep_on=cb_primary.state_dependencies)

        # Bootstrap state
        with CodeBuilder(label="bootstrap") as cb_bootstrap:
            self.emit_rk_startup(cb_bootstrap)
            self.emit_epilogue(cb_bootstrap)
            with cb_bootstrap.if_(self.step, "==", bootstrap_steps):
                cb_bootstrap.state_transition("primary")

        states = {}
        states["initialization"] = TimeIntegratorState.from_cb(cb_init, "bootstrap")
        states["bootstrap"] = TimeIntegratorState.from_cb(cb_bootstrap, "bootstrap")
        states["primary"] = TimeIntegratorState.from_cb(cb_primary, "primary")

        return TimeIntegratorCode(
            instructions=cb_init.instructions | cb_bootstrap.instructions |
            cb_primary.instructions,
            states=states,
            initial_state="initialization")


class MRABCodeEmitter(MRABProcessor):

    def __init__(self, stepper, cb, y, t, rhss):
        MRABProcessor.__init__(self, stepper.method, stepper.substep_count)
        self.stepper = stepper
        self.cb = cb
        self.t_start = t

        # Mapping from method variable names to code variable names
        self.name_to_variable = {}

        self.context = {}
        self.var_time_level = {}

        # Names of instructions that were generated in the previous step
        self.last_step = []

        self.rhss = rhss

        y_fast, y_slow = y
        from leap.method.ab.multirate.methods import CO_FAST, CO_SLOW
        self.last_y = {CO_FAST: y_fast, CO_SLOW: y_slow}

        self.hist_head_time_level = dict((hn, 0) for hn in HIST_NAMES)

    def get_variable(self, name):
        """Return a variable for a name found in the method description."""

        if name not in self.name_to_variable:
            from string import ascii_letters
            from pymbolic import var
            prefix = "".join([c for c in name if c in ascii_letters])
            self.name_to_variable[name] = var(self.cb.fresh_var_name(prefix))
        return self.name_to_variable[name]

    def run(self):
        super(MRABCodeEmitter, self).run()

        # Update the slow and fast components.
        from leap.method.ab.multirate.methods import CO_FAST, CO_SLOW
        self.cb(self.last_y[CO_SLOW], self.context[self.method.result_slow])
        self.cb(self.last_y[CO_FAST], self.context[self.method.result_fast])

    def integrate_in_time(self, insn):
        from leap.method.ab.multirate.methods import CO_FAST
        from leap.method.ab.multirate.methods import \
            HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S
        from pymbolic import var

        if insn.component == CO_FAST:
            self_hn, cross_hn = HIST_F2F, HIST_S2F
        else:
            self_hn, cross_hn = HIST_S2S, HIST_F2S

        # Compute AB coefficients

        start_time_level = self.eval_expr(insn.start)
        end_time_level = self.eval_expr(insn.end)

        levels_self = numpy.arange(0, -self.stepper.orders[self_hn], -1, dtype=numpy.float64)
        levels_cross = numpy.arange(0, -self.stepper.orders[cross_hn], -1, dtype=numpy.float64)

        self.cb("timestep", self.stepper.large_dt)
        self.cb.fence()

        self.cb("n_cross",len(levels_cross))
        self.cb("n_self",len(levels_self))

        if self.stepper.hist_is_fast[self_hn]:
            levels_self /= self.substep_count

        levels_self += self.hist_head_time_level[self_hn]/self.substep_count

        if self.stepper.hist_is_fast[cross_hn]:
            levels_self /= self.substep_count

        levels_cross += self.hist_head_time_level[cross_hn]/self.substep_count

        t_start = start_time_level / self.substep_count
        t_end = end_time_level / self.substep_count

        self.cb.fence()
        self.cb("levels_cross","`<builtin>array`(n_cross)")
        self.cb("levels_self","`<builtin>array`(n_self)")
        self.cb.fence()

        self.cb("start_time", t_start)
        self.cb("end_time", t_end)
        self.cb.fence()

        for i in range(len(levels_self)):
            self.cb("levels_self[{0}]".format(i), levels_self[i])
            self.cb.fence()

        for i in range(len(levels_cross)):
            self.cb("levels_cross[{0}]".format(i), levels_cross[i])
            self.cb.fence()

        self.cb("point_eval_vec_cross", "`<builtin>array`(n_cross)")
        self.cb("point_eval_vec_self", "`<builtin>array`(n_self)")

        self.cb("vdm_transpose_cross", "`<builtin>array`(n_cross*n_cross)")
        self.cb("vdm_transpose_self", "`<builtin>array`(n_self*n_self)")
        self.cb.fence()

        self.cb("point_eval_vec_cross[g]", "1 / (g + 1) * (end_time ** (g + 1)- start_time ** (g + 1)) ",
                loops=[("g", 0, "n_cross")])
        self.cb("point_eval_vec_self[g]", "1 / (g + 1) * (end_time ** (g + 1)- start_time ** (g + 1)) ",
                loops=[("g", 0, "n_self")])
        self.cb("vdm_transpose_cross[g*n_cross + h]", "levels_cross[g]**h",
                loops=[("g", 0, "n_cross"), ("h", 0, "n_cross")])
        self.cb("vdm_transpose_self[g*n_self + h]", "levels_self[g]**h",
                loops=[("g", 0, "n_self"), ("h", 0, "n_self")])

        self.cb.fence()

        self.cb("new_cross_coeffs", "`<builtin>linear_solve`(vdm_transpose_cross, point_eval_vec_cross, n_cross, 1)")
        self.cb("new_self_coeffs", "`<builtin>linear_solve`(vdm_transpose_self, point_eval_vec_self, n_self, 1)")

        self.cb.fence()

        # We can complete the integration by then performing the necessary linear combination, again using new built-ins

        if start_time_level == 0 or (insn.result_name not in self.context):
            my_y = self.last_y[insn.component]
            assert start_time_level == 0
        else:
            my_y = self.context[insn.result_name]
            assert start_time_level == self.var_time_level[insn.result_name]

        # Define the self and cross histories

        hists = self.stepper.histories
        self_history = hists[self_hn][:]
        cross_history = hists[cross_hn][:]

        # Define a Python-side vector for the calculated coefficients (will be used with linear_comb)

        new_self_coeffs_pyvar = var("newself")
        new_cross_coeffs_pyvar = var("newcross")

        self.cb.fence()

        new_self_coeffs_py = [new_self_coeffs_pyvar[i] for i in range(len(levels_self))]
        new_cross_coeffs_py = [new_cross_coeffs_pyvar[i] for i in range(len(levels_cross))]

        # Use loops to assign each element of this vector to an element from our newly calculated coeff vector (Fortran-side)

        self.cb("newself","`<builtin>array`(n_self)")
        self.cb("newcross","`<builtin>array`(n_cross)")
        self.cb.fence()

        for i in range(len(levels_self)):
            self.cb(new_self_coeffs_py[i], "new_self_coeffs[{0}]".format(i))
            self.cb.fence()

        for i in range(len(levels_cross)):
            self.cb(new_cross_coeffs_py[i], "new_cross_coeffs[{0}]".format(i))
            self.cb.fence()

        needs_fence = insn.result_name in self.name_to_variable
        new_y_var = self.get_variable(insn.result_name)

        if needs_fence:
            self.cb.fence()

        # Perform the linear combination to obtain our new_y

        self.cb(new_y_var, my_y + self.stepper.large_dt * (linear_comb(new_cross_coeffs_py, cross_history) + linear_comb(new_self_coeffs_py, self_history)))
        self.cb.fence()

        self.context[insn.result_name] = new_y_var
        self.var_time_level[insn.result_name] = end_time_level

        MRABProcessor.integrate_in_time(self, insn)

    def history_update(self, insn):
        time_slow = self.var_time_level[insn.slow_arg]

        t = (self.t_start
                + self.stepper.large_dt*time_slow/self.stepper.substep_count)

        rhs = self.rhss[HIST_NAMES.index(insn.which)]

        hist = self.stepper.histories[insn.which]

        reverse_hist = hist[::-1]

        # Move all the histories by one step forward
        for h, h_next in zip(reverse_hist, reverse_hist[1:]):
            self.cb.fence()
            self.cb(h, h_next)

        time_hist = self.stepper.time_histories[insn.which]

        reverse_time_hist = time_hist[::-1]

        # Move time histories by one step forward
        for th, th_next in zip(reverse_time_hist, reverse_time_hist[1:]):
            self.cb.fence()
            self.cb(th, th_next)

        # Tack on the new time
        self.cb.fence()
        self.cb(time_hist[0], t)

        # Compute the new RHS
        self.cb.fence()
        self.cb(hist[0], rhs(t=t, f=self.context[insn.fast_arg],
                             s=self.context[insn.slow_arg]))

        if self.stepper.hist_is_fast[insn.which]:
            self.hist_head_time_level[insn.which] += 1
        else:
            self.hist_head_time_level[insn.which] += self.stepper.substep_count

        MRABProcessor.history_update(self, insn)
