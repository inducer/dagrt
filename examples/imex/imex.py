"""This example illustrates the use of a custom solver."""

import numpy as np


class KapsProblem(object):
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

    def jacobian(self, t, y):
        y_2 = y[1]
        return np.array([
            [-self._epsilon_inv - 2, 2 * self._epsilon_inv * y_2],
            [1, -1 - 2 * y_2]
        ])

    def exact(self, t):
        return np.array([np.exp(-2 * t), np.exp(-t)])


_atol = 1.0e-3


def solver(f, j, t, u_n, x, c):
    """Kennedy and Carpenter, page 15"""
    import numpy.linalg as nla
    I = np.eye(len(u_n))
    u = u_n
    while True:
        M = I - c * j(t, u)
        r = -(u - u_n) + x + c * f(t=t, y=u)
        d = nla.solve(M, r)
        u = u + d
        if 0.005 * _atol >= nla.norm(d):
            return f(t=t, y=u)


def solver_hook(expression, solve_component, guess, template=None):
    """The solver hook returns an expression that will be used to solve for the
    implicit component.
    """
    from leap.vm.expression import match, substitute

    # Match the expression with the template.
    assert template
    template = substitute(template, {"solve_component": solve_component})
    subst = match(template, expression, ["sub_y", "coeff", "t"])

    # Match the components that were found.
    from pymbolic import var
    u_n = var("<state>y")
    f = var("<func>impl_y")
    j = var("<func>j")
    t = subst["t"]
    x = subst["sub_y"]
    c = subst["coeff"]

    # Return the expression that calls the solver.
    return var("<func>solver")(f, j, t, u_n, x, c)


def run():
    from functools import partial
    from leap.method.rk.imex import KennedyCarpenterIMEXARK4
    from leap.vm.codegen import PythonCodeGenerator

    # Construct the method generator.
    mgen = KennedyCarpenterIMEXARK4("y", atol=_atol)

    # Generate the code for the method.
    template = mgen.implicit_expression()[0]
    print("Expression for solver: " + str(template))
    sgen = partial(solver_hook, template=template)
    code = mgen.generate(sgen)
    IMEXIntegrator = PythonCodeGenerator("IMEXIntegrator").get_class(code)

    # Set up the problem and run the method.
    problem = KapsProblem(epsilon=0.001)
    integrator = IMEXIntegrator(function_map={
        mgen.rhs_expl_func.name: problem.nonstiff,
        mgen.rhs_impl_func.name: problem.stiff,
        "<func>solver": solver,
        "<func>j": problem.jacobian})

    integrator.set_up(t_start=problem.t_start,
                      dt_start=1.0e-1,
                      context={"y": problem.initial()})

    t = None
    y = None

    for event in integrator.run(t_end=problem.t_end):
        if isinstance(event, integrator.StateComputed):
            t = event.t
            y = event.state_component

    print("Error: " + str(np.linalg.norm(y - problem.exact(t))))


if __name__ == "__main__":
    run()
