# -*- coding: utf-8 -*-

"""Multirate-AB ODE solver."""

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
from pytools import memoize_method
from leap.method.ab import AdamsBashforthTimeStepperBase
from leap.method.ab.utils import make_generic_ab_coefficients, linear_comb
from leap.method.ab.multirate.methods import (HIST_NAMES, HIST_F2F, HIST_S2F,
                                              HIST_F2S, HIST_S2S)
from leap.method.ab.multirate.processors import MRABProcessor
from leap.vm.language import SimpleCodeBuilder
from pymbolic import var


__doc__ = """
.. autoclass:: TwoRateAdamsBashforthTimeStepper
"""


class TwoRateAdamsBashforthTimeStepper(AdamsBashforthTimeStepperBase):
    """Simultaneously timesteps two parts of an ODE system,
    the first with a small timestep, the second with a large timestep.

    [1] C.W. Gear and D.R. Wells, "Multirate linear multistep methods," BIT
    Numerical Mathematics,  vol. 24, Dec. 1984,pg. 484-502.
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
        # States used in RK initialization
        self.rk_rhss = {
            HIST_F2F: var('<p>rk_f2f'),
            HIST_S2F: var('<p>rk_s2f'),
            HIST_F2S: var('<p>rk_f2s'),
            HIST_S2S: var('<p>rk_s2s')
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

        for hn in HIST_NAMES:
            hname = hn().__class__.__name__
            vnames = [var('<p>last_rhs_%s_%d' % (hname, i)) for i in
                      range(self.orders[hn])]
            self.histories[hn] = vnames

        self.hist_is_fast = {
                HIST_F2F: True,
                HIST_S2F: self.method.s2f_hist_is_fast,
                HIST_S2S: False,
                HIST_F2S: False
                }

    def emit_initialization(self, cbuild):
        """Initialize method variables. Returns the initialization list."""

        # Initial value of RK y vector
        with SimpleCodeBuilder(cbuild) as builder:
            builder.assign(self.step, 0)

        # Initial value of RK derivatives
            for hist_component, function in self.component_functions.items():
                assignee = self.rk_rhss[hist_component]
                builder.assign(assignee,
                               function(t=self.t, s=self.slow, f=self.fast))

        return builder.last_added_instruction_id

    def emit_small_rk_step(self, builder, t, name_prefix):
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
                    builder.assign(stage_rhss[component][stage_number],
                                   self.rk_rhss[component])
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
                    builder.assign(stage_rhss[component][stage_number],
                                   function(t=t + c * self.small_dt,
                                            s=stage_s, f=stage_f))

        builder.assign(self.slow, self.slow + self.small_dt *
                       sum(coeff * (stage_rhss[HIST_F2S][k] +
                                    stage_rhss[HIST_S2S][k])
                       for k, coeff in enumerate(rk_coeffs)))

        builder.assign(self.fast, self.fast + self.small_dt *
                       sum(coeff * (stage_rhss[HIST_F2F][k] +
                                    stage_rhss[HIST_S2F][k])
                       for k, coeff in enumerate(rk_coeffs)))

        for hist_component, function in self.component_functions.items():
            assignee = self.rk_rhss[hist_component]
            builder.assign(assignee, function(t=t + self.small_dt,
                                    s=self.slow, f=self.fast))

    def emit_startup(self, cbuild):
        """Initializes the stepper with an RK method.
        Returns the code that computes the startup history."""

        from pymbolic import var
        from pymbolic.primitives import Comparison

        initialization_steps = self.max_order - 1
        total_substeps = initialization_steps * self.substep_count

        make_step_history = lambda prefix: \
            [var(prefix + str(i)) for i in range(total_substeps)]

        rk_rhs_history = {
            HIST_F2F: make_step_history('<p>rk_f2f_'),
            HIST_S2F: make_step_history('<p>rk_s2f_'),
            HIST_F2S: make_step_history('<p>rk_f2s_'),
            HIST_S2S: make_step_history('<p>rk_s2s_')
            }

        with SimpleCodeBuilder(cbuild) as builder:
            for i in range(initialization_steps):
                # Add code to run the current step.
                with builder.condition(Comparison(self.step, "==", i)):
                    # Add code to run the current substep.
                    for j in range(self.substep_count):
                        history_index = i * self.substep_count + j
                        # Save the RHS components to the history.
                        for rhs_component in rk_rhs_history:
                            builder.assign(
                                rk_rhs_history[rhs_component][history_index],
                                self.rk_rhss[rhs_component])
                        # Add the next step of the RK initialization.
                        name_prefix = 'substep{i}_{j}'.format(i=i, j=j)
                        self.emit_small_rk_step(builder,
                                                self.t + j * self.small_dt,
                                                name_prefix)

            # Once the entire startup history has been computed with the RK
            # method, the saved RK components are mapped to the appropriate
            # AB components.
            with builder.condition(Comparison(
                    self.step, "==", initialization_steps)):
                for component in HIST_NAMES:
                    history = rk_rhs_history[component] + \
                        [self.rk_rhss[component]]
                    history.reverse()
                    if not self.hist_is_fast[component]:
                        history = history[::self.substep_count]
                    history = history[:self.orders[component]]
                    for index, entry in enumerate(history):
                        builder.assign(self.histories[component][index], entry)
                    assert len(self.histories[component]) == \
                        self.orders[component]

        return builder.last_added_instruction_id

    def emit_ab_method(self, cbuild):
        """Add code for the main Adams-Bashforth method."""
        rhss = [self.f2f, self.s2f, self.f2s, self.s2s]
        codegen = MRABCodeEmitter(self, cbuild, (self.fast, self.slow), self.t, rhss)
        codegen.run()
        return codegen.get_instructions()

    def emit_main_branches(self, cbuild, startup, main_code):
        """Add code that determines whether to perform initialization or step into
        the main method."""

        from leap.vm.language import If, AssignExpression
        from pymbolic.primitives import Comparison

        incr_step = cbuild.add_and_get_ids(
            AssignExpression(self.step.name, self.step + 1))
        branches = cbuild.add_and_get_ids(
            If(condition=Comparison(self.step, "<", self.max_order),
               then_depends_on=startup + incr_step,
               else_depends_on=[], id='rk_branch'),
            If(condition=Comparison(self.step, "==", self.max_order),
               then_depends_on=main_code, else_depends_on=[],
               depends_on=['rk_branch']))
        cbuild.commit()
        return branches

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

    def emit_epilogue(self, cbuild, glue):
        with SimpleCodeBuilder(cbuild, glue) as builder:
            builder.yield_state(self.slow, "slow", self.t + self.dt, "final")
            builder.yield_state(self.fast, "fast", self.t + self.dt, "final")
            builder.assign(self.t, self.t + self.dt)

        return builder.last_added_instruction_id

    def __call__(self):
        from leap.vm.language import CodeBuilder, TimeIntegratorCode

        cbuild = CodeBuilder()
        initialization = self.emit_initialization(cbuild)
        startup = self.emit_startup(cbuild)
        main_code = self.emit_ab_method(cbuild)
        glue = self.emit_main_branches(cbuild, startup, main_code)
        epilogue = self.emit_epilogue(cbuild, glue)

        return TimeIntegratorCode(instructions=cbuild.instructions,
                                  initialization_dep_on=initialization,
                                  step_dep_on=epilogue,
                                  step_before_fail=False)


class MRABCodeEmitter(MRABProcessor):

    def __init__(self, stepper, cbuild, y, t, rhss):
        MRABProcessor.__init__(self, stepper.method, stepper.substep_count)
        self.stepper = stepper
        self.cbuild = cbuild
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
            self.name_to_variable[name] = \
                var(self.cbuild.fresh_var_name(prefix))
        return self.name_to_variable[name]

    def run(self):
        super(MRABCodeEmitter, self).run()

        # Update the slow and fast components.
        from leap.method.ab.multirate.methods import CO_FAST, CO_SLOW
        from leap.vm.language import AssignExpression

        self.last_step = self.cbuild.add_and_get_ids(
            AssignExpression(self.last_y[CO_SLOW].name,
            self.context[self.method.result_slow], depends_on=self.last_step),
            AssignExpression(self.last_y[CO_FAST].name,
            self.context[self.method.result_fast], depends_on=self.last_step))

        self.cbuild.commit()

    def integrate_in_time(self, insn):
        from leap.vm.language import AssignExpression

        from leap.method.ab.multirate.methods import CO_FAST
        from leap.method.ab.multirate.methods import \
            HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S

        if insn.component == CO_FAST:
            self_hn, cross_hn = HIST_F2F, HIST_S2F
        else:
            self_hn, cross_hn = HIST_S2S, HIST_F2S

        start_time_level = self.eval_expr(insn.start)
        end_time_level = self.eval_expr(insn.end)

        self_coefficients = self.stepper.get_coefficients(
            self.stepper.hist_is_fast[self_hn],
            self.hist_head_time_level[self_hn],
            start_time_level, end_time_level,
            self.stepper.orders[self_hn])
        cross_coefficients = self.stepper.get_coefficients(
            self.stepper.hist_is_fast[cross_hn],
            self.hist_head_time_level[cross_hn],
            start_time_level, end_time_level,
            self.stepper.orders[cross_hn])

        if start_time_level == 0 or (insn.result_name not in self.context):
            my_y = self.last_y[insn.component]
            assert start_time_level == 0
        else:
            my_y = self.context[insn.result_name]
            assert start_time_level == self.var_time_level[insn.result_name]

        hists = self.stepper.histories
        self_history = hists[self_hn][:]
        cross_history = hists[cross_hn][:]

        my_new_y = my_y + self.stepper.large_dt * (
                linear_comb(self_coefficients, self_history)
                + linear_comb(cross_coefficients, cross_history))

        new_y_var = self.get_variable(insn.result_name)

        new_y = self.cbuild.add_and_get_ids(AssignExpression(new_y_var.name,
                    my_new_y, depends_on=self.last_step))

        self.last_step = new_y

        self.cbuild.commit()

        self.context[insn.result_name] = new_y_var
        self.var_time_level[insn.result_name] = end_time_level

        MRABProcessor.integrate_in_time(self, insn)

    def history_update(self, insn):
        from leap.vm.language import AssignExpression

        time_slow = self.var_time_level[insn.slow_arg]

        t = (self.t_start
                + self.stepper.large_dt*time_slow/self.stepper.substep_count)

        rhs = self.rhss[HIST_NAMES.index(insn.which)]

        hist = self.stepper.histories[insn.which]

        reverse_hist = hist[::-1]

        # Move all the histories by one step forward
        assignments = []
        for h, h_next in zip(reverse_hist, reverse_hist[1:]):
            assignments += self.cbuild.add_and_get_ids(AssignExpression(h.name,
                                h_next, depends_on=self.last_step + assignments))

        # Compute the new RHS
        assignments += self.cbuild.add_and_get_ids(
            AssignExpression(assignee=hist[0].name,
                expression=rhs(t=t, f=self.context[insn.fast_arg],
                               s=self.context[insn.slow_arg]),
                depends_on=self.last_step + assignments))

        self.last_step = assignments

        self.cbuild.commit()

        if self.stepper.hist_is_fast[insn.which]:
            self.hist_head_time_level[insn.which] += 1
        else:
            self.hist_head_time_level[insn.which] += self.stepper.substep_count

        MRABProcessor.history_update(self, insn)

    def get_instructions(self):
        return self.last_step
