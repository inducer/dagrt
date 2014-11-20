"""Multirate-AB ODE solver."""

from __future__ import division

__copyright__ = """
Copyright (C) 2009 Andreas Stock, Andreas Kloeckner
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


class MRABProcessor(object):
    def __init__(self, method, substep_count):
        self.method = method
        self.substep_count = substep_count
        self.substep_loop_start = 0

    def integrate_in_time(self, insn):
        self.insn_counter += 1

    def history_update(self, insn):
        self.insn_counter += 1

    def start_substep_loop(self, insn):
        self.insn_counter += 1
        self.substep_loop_start = self.insn_counter

    def eval_expr(self, expr):
        from pymbolic import evaluate_kw
        return evaluate_kw(expr,
                substep_index=self.substep_index,
                substep_count=self.substep_count)

    def end_substep_loop(self, insn):
        self.substep_index += 1
        if self.substep_index >= self.eval_expr(insn.loop_end):
            self.insn_counter += 1
        else:
            self.insn_counter = self.substep_loop_start

    def run(self):
        self.insn_counter = 0
        self.substep_index = 0

        while not self.insn_counter >= len(self.method.steps):
            self.method.steps[self.insn_counter].visit(self)
