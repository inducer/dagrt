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
from leap.method.ab.multirate.methods import HIST_NAMES
from leap.method.ab.multirate.processors import MRABProcessor


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
        # Coupled function from that returns all four RHSs
        self.coupled = var('<func>coupled')
        # Individual component functions
        self.f2f = var('<func>f2f')
        self.s2f = var('<func>s2f')
        self.s2s = var('<func>s2s')
        self.f2s = var('<func>f2s')
        # Coupled states for RK initialization
        self.rk_y = var('<p>rk_y')
        self.rk_rhs = var('<p>rk_rhs')

        self.large_dt = self.dt
        self.small_dt = self.dt / substep_count
        self.substep_count = substep_count

        from leap.method.ab.multirate.methods import \
            HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S

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

        from leap.vm.language import AssignExpression

        add_and_get_ids = cbuild.add_and_get_ids
        initialization = []

        # Initial value of step
        initialization += add_and_get_ids(AssignExpression(self.step.name, 0))

        # Initial value of RK y vector
        yval, = add_and_get_ids(AssignExpression(self.rk_y.name,
            numpy.array((self.fast, 0., self.slow, 0.), dtype='object')))
        initialization.append(yval)

        # Initial value of RK derivative matrix
        initialization += add_and_get_ids(
            AssignExpression(assignee=self.rk_rhs.name,
                expression=self.coupled(t=self.t, y=self.rk_y),
                depends_on=[yval]))
        cbuild.commit()

        return initialization

    def emit_startup(self, cbuild):
        """Initializes the stepper with an RK method.
        Returns the code that computes the startup history."""

        from leap.vm.language import AssignExpression, If
        add_and_get_ids = cbuild.add_and_get_ids
        from pymbolic import var
        from pymbolic.primitives import Comparison

        steps = self.max_order * self.substep_count

        rhs_history = dict([(i, var('<p>rhs_y_%d' % i)) for i in range(steps)])

        startup_conditions = []

        # Note: both y and RHS need to be 2x2 matrices and functions need to
        # be adapted to this convention. Yuck
        component_id = self.coupled.name

        for i in range(self.max_order):
            rk = []

            # Add code to run the current substeps.
            for j in range(self.substep_count):
                history_index = i * self.substep_count + j
                # Save the RHS to the history
                save_vals = add_and_get_ids(AssignExpression(
                    rhs_history[history_index].name, self.rk_rhs,
                    depends_on=rk))
                # Compute the new value of y and rhs
                rk += self.emit_rk_body(cbuild, component_id,
                    self.max_order, self.rk_y, self.rk_rhs, self.t
                    + j * self.small_dt, self.small_dt, depends_on=save_vals,
                    label=('substep%d_%d' % (i, j)))

            # Unpack the y values into slow and fast components.
            rk += add_and_get_ids(AssignExpression(self.fast.name,
                    self.rk_y.index(0) + self.rk_y.index(1), depends_on=rk),
                    AssignExpression(self.slow.name,
                    self.rk_y.index(2) + self.rk_y.index(3), depends_on=rk))
            # Add code to run the current step based on the value of <p>step
            if i < self.max_order - 1:
                last_step, = add_and_get_ids(If(
                    condition=Comparison(self.step, "==", i),
                    then_depends_on=rk, else_depends_on=[]))
                startup_conditions.append(last_step)

        cbuild.commit()

        # Add code that, once the startup history has been created, will
        # initialize the components to appropriate values.

        rhs_hist = [rhs_history[i] for i in range(steps)] + [self.rk_rhs]
        rhs_hist.reverse()

        create_histories = []

        # RHSs are 2x2 matrices containing history entries
        for i, hn in enumerate(HIST_NAMES):
            hist = rhs_hist
            if not self.hist_is_fast[hn]:
                hist = hist[::self.substep_count]
            hist = hist[:self.orders[hn]]
            for j, entry in enumerate(hist):
                create_histories += add_and_get_ids(AssignExpression(
                    self.histories[hn][j].name, entry.index(i),
                    depends_on=rk))
            assert len(self.histories[hn]) == self.orders[hn]

        startup_conditions += add_and_get_ids(
            If(condition=Comparison(self.step, "==", self.max_order - 1),
            then_depends_on=create_histories, else_depends_on=[]))

        cbuild.commit()

        return startup_conditions

    def emit_ab_method(self, cbuild):
        """Add code for the main Adams-Bashforth method."""
        rhss = [self.f2f, self.s2f, self.f2s, self.s2s]
        codegen = MRABCodeEmitter(self, cbuild, (self.fast, self.slow), self.t, rhss)
        codegen.run()
        return codegen.get_instructions()

    def emit_main_branch(self, cbuild, startup, main_code):
        """Add code that determines whether to perform initialization or step into
        the main method."""

        from leap.vm.language import If, AssignExpression
        from pymbolic.primitives import Comparison

        incr_step = cbuild.add_and_get_ids(
            AssignExpression(self.step.name, self.step + 1))
        branch = cbuild.add_and_get_ids(If(condition=Comparison(self.step,
            "<", self.max_order), then_depends_on=startup + incr_step,
            else_depends_on=main_code))
        cbuild.commit()
        return branch

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

    def __call__(self):
        from leap.vm.language import CodeBuilder, TimeIntegratorCode

        cbuild = CodeBuilder()
        initialization = self.emit_initialization(cbuild)
        startup = self.emit_startup(cbuild)
        ab_method = self.emit_ab_method(cbuild)
        glue = self.emit_main_branch(cbuild, startup, ab_method)
        epilogue = self.emit_epilogue(cbuild,
            numpy.array((self.fast, self.slow), dtype='object'), '', glue)
        cbuild.commit()

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
            prefix = filter(lambda x: x in ascii_letters, name)
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
