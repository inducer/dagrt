"""Collection of ODE systems partitioned into stiff and nonstiff
components"""

__copyright__ = """
Copyright (C) 2014 Andreas Kloeckner
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

import numpy as np


class StiffPartitionedODESystem(object):
    """
    .. attribute :: t_start
    .. attribute :: t_end
    """

    def initial(self):
        """Return an initial condition."""
        raise NotImplementedError()

    def nonstiff(self, t, y):
        """Return the nonstiff component."""
        raise NotImplementedError()

    def stiff(self, t, y):
        """Return the stiff component."""
        raise NotImplementedError()

    def exact(self, t):
        """Return the exact solution, if available."""
        raise NotImplementedError()

    def __call__(self, t, y):
        return self.nonstiff(t, y) + self.stiff(t, y)


class KapsProblem(StiffPartitionedODESystem):
    """
    From Kennedy and Carpenter, Section 7.1

    y_1' = - (epsilon^{-1} + 2) y_1 + epsilon^{-1} y_2^2
    y_2' = y_1 - y_2 - y_2^2

    0 <= t <= 1

    The initial conditions are

      y_1 = y_2 = 1.

    The exact solution is

      y_1 = exp(-2t)
      y_2 = exp(-t).

    The stiff component are the terms multiplied by epsilon^{-1}.
    """

    def __init__(self, epsilon):
        self._epsilon_inv = 1 / epsilon
        self.t_start = 0
        self.t_end = 1

    def initial(self):
        return np.array([1., 1.])

    def nonstiff(self, t, y):
        y_1 = y[0]
        y_2 = y[1]
        return np.array([-2 * y_1, y_1 - y_2 - y_2 ** 2])

    def stiff(self, t, y):
        y_1 = y[0]
        y_2 = y[1]
        return np.array([-self._epsilon_inv * (y_1 - y_2 ** 2), 0])

    def exact(self, t):
        return np.array([np.exp(-2 * t), np.exp(-t)])


class VanDerPolProblem(StiffPartitionedODESystem):

    def __init__(self, mu=30):
        self._mu = mu
        self.t_start = 0
        self.t_end = 100

    def initial(self):
        return np.array([2, 0], dtype=np.float64)

    def nonstiff(self, t, y):
        return np.array([y[1], 0])

    def stiff(self, t, y):
        u1 = y[0]
        u2 = y[1]
        return np.array([0, -self._mu * (u1 ** 2 - 1) * u2 - u1])
