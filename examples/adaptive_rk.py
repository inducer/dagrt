#!/usr/bin/env python
"""Example of implementing a simple adaptive Runge-Kutta method."""

__copyright__ = "Copyright (C) 2015 Matt Wala"

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

from dagrt.language import DAGCode, CodeBuilder
from pymbolic import var


def adaptive_rk_method(tol):
    """
    The difference between the results of Euler's method
       y_e = y_n + h f(t_n, y_n)
    and Heun's method
       y_h = y_n + h / 2 (f(t_n, y_n), f(t_n + dt, y_e))
    can be used as an estimate of the high order error term in Euler's method.

    This allows us to adapt the time step so that the local error falls within a
    specified tolerance.
    """

    # Set up variables
    y = var("<state>y")
    g = var("<func>g")
    y_e = var("y_e")
    y_h = var("y_h")
    dt = var("<dt>")
    t = var("<t>")
    dt_old = var("dt_old")

    # Helpers for expression fragments
    def norm(val):
        return var("<builtin>norm_2")(val)

    def dt_scaling(tol, err):
        # Return a suitable scaling factor for dt.
        # Ensure to guard against excessive increase or divide-by-zero.
        from pymbolic.primitives import Max, Min
        return Min(((tol / Max((1.0e-16, norm(err)))) ** (1 / 2), 2.0))

    # Code for the main state
    with CodeBuilder("adaptrk") as cb:
        # Euler
        cb(y_e, y + dt * g(t, y))

        # Heun
        cb(y_h, y + dt / 2 * (g(t, y) + g(t + dt, y_e)))

        # Adaptation
        cb(dt_old, dt)
        cb(dt, dt * dt_scaling(tol, y_h - y_e))

        # Accept or reject step
        with cb.if_(norm(y_h - y_e), ">=", tol):
            cb.fail_step()
        with cb.else_():
            cb(y, y_h)
            cb(t, t + dt_old)

    return DAGCode.from_phases_list(
            [cb.as_execution_phase("adaptrk")], "adaptrk")


def main():
    def rhs(t, y):
        u, v = y
        return np.array([v, -u/t**2], dtype=np.float64)

    def soln(t):
        inner = np.sqrt(3)/2*np.log(t)
        return np.sqrt(t)*(
                5*np.sqrt(3)/3*np.sin(inner)
                + np.cos(inner)
        )

    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator("AdaptiveRK")

    tolerances = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-5]
    errors = []

    for tol in tolerances:
        method = adaptive_rk_method(tol)
        AdaptiveRK = codegen.get_class(method)  # noqa: N806
        solver = AdaptiveRK({"<func>g": rhs})
        solver.set_up(t_start=1.0, dt_start=0.1, context={"y": np.array([1., 3.])})
        for evt in solver.run(t_end=10.0):
            final_time = evt.t
        errors.append(np.abs(solver.global_state_y[0] - soln(final_time)))

    print("Tolerance\tError")
    print("-" * 25)
    for tol, error in zip(tolerances, errors):
        print(f"{tol:.2e}\t{error:.2e}")


if __name__ == "__main__":
    main()
