"""
Solve the advection equation

  u_t = a u_x
  u(0) = init

where x lies in [0, 2]. The solution uses a multirate AB method and the method
of lines. The x coordinates are discretized so that [1, 2] is more densely
partitioned than [0, 1] by a given ratio.
"""


from __future__ import division

from leap.vm.codegen import PythonCodeGenerator
from leap.vm.exec_numpy import StateComputed
from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
import leap.method.ab.multirate.methods

import numpy as np
import scipy.linalg as sla
from mpl_toolkits.mplot3d import axes3d  # noqa
import matplotlib.pyplot as plt
from pytools import DictionaryWithDefault


FastestFirst = leap.method.ab.multirate.methods.methods['Fq']


def make_rhs(matrix, multiply_fast_component):
    """Return a function suitable for use as an RHS that does matrix
    multiplication. If multiply_fast_component is True, then the fast component
    is multiplied, otherwise the slow component is multiplied.
    """
    def f(t, u, v):
        return matrix.dot(u if multiply_fast_component else v)
    return f


def make_coupled(f2f, f2s, s2f, s2s):
    """Return a function that couples the RHSs in a manner that is suitable to
    be passed as input to the MRAB class. This coupling is used during the
    initialization stage of the method. Eventually, this might be hidden from
    the user."""
    def coupled(t, y):
        args = (t, y[0] + y[1], y[2] + y[3])
        return np.array((f2f(*args), f2s(*args), s2f(*args), s2s(*args)),)
    return coupled


# The function below return the right hand sides corresponding to the different
# components.


def make_f2f(a, nx_fast, dx_fast, ratio):
    col = np.zeros(nx_fast)
    row = np.zeros(nx_fast)
    col[0] = -1
    row[0:2] = [-1, 1]
    matrix = sla.toeplitz(col, row)
    matrix *= a / dx_fast
    return make_rhs(matrix, True)


def make_s2f(a, nx_fast, nx_slow, dx_fast, ratio):
    return make_rhs(np.zeros((nx_fast, nx_slow),), False)


def make_f2s(a, nx_fast, nx_slow, dx_fast, ratio):
    matrix = np.zeros((nx_slow, nx_fast),)
    matrix[nx_slow - 1, 0] = a / dx_fast
    return make_rhs(matrix, True)


def make_s2s(a, nx_slow, dx_fast, ratio):
    col = np.zeros(nx_slow)
    row = np.zeros(nx_slow)
    col[0] = -1
    row[0:2] = [-1, 1]
    matrix = sla.toeplitz(col, row)
    matrix = (a / (dx_fast * ratio)) * sla.toeplitz(col, row)
    matrix[nx_slow - 1, nx_slow - 1] = (-a / dx_fast)
    return make_rhs(matrix, False)


def make_multirate_initial_conditions(dx_fast, ratio):
    """Return a pair consisting of the initial x space and the initial values
    of the function on the x space."""
    nx_slow = int(1 / (dx_fast * ratio))
    nx_fast = int(1 / dx_fast)
    slow_space = np.linspace(0, 1, nx_slow)
    fast_space = 1 + np.linspace(dx_fast, 1, nx_fast)
    space = np.concatenate((slow_space, fast_space),)
    init = lambda x: -np.sin(x * np.pi) if x >= 1.0 else 0.0
    return (space, np.vectorize(init)(space))


def run_multirate_method(method, y_fast, y_slow, dt, t_start, t_end):
    """Run the given method and return the history."""
    method.set_up(t_start=t_start, dt_start=dt, state={'fast': y_fast,
                                                       'slow': y_slow})
    method.initialize()
    history = [np.concatenate((y_slow, y_fast),)]
    for event in method.run(t_end=t_end):
        if isinstance(event, StateComputed):
            slow = event.state_component[1]
            fast = event.state_component[0]
            history.append(np.concatenate((slow, fast),))
    return np.array(history)


def make_multirate_method(f2f, s2f, f2s, s2s, ratio=2, order=3):
    """Return the object that drives the multirate method for the given
    parameters."""
    orders = DictionaryWithDefault(lambda x: order)
    code = TwoRateAdamsBashforthTimeStepper(FastestFirst, orders, ratio)()
    MRABMethod = PythonCodeGenerator(class_name='MRABMethod').get_class(code)

    rhs_map = {'<func>f2f': f2f, '<func>s2f': s2f, '<func>f2s': f2s,
               '<func>s2s': s2s,
               '<func>coupled': make_coupled(f2f, s2f, f2s, s2s)}

    return MRABMethod(rhs_map)


def plot_multirate_initial_conditions(dx, ratio):
    figure, axis = plt.subplots()
    axis.plot(*make_multirate_initial_conditions(dx, ratio))
    plt.show()


def plot_multirate_history(a, dx_fast, ratio, dt, t_start, t_end):
    nx_slow = int(1 / (dx_fast * ratio))
    nx_fast = int(1 / dx_fast)
    # Make the components.
    f2f = make_f2f(a, nx_fast, dx_fast, ratio)
    s2f = make_s2f(a, nx_fast, nx_slow, dx_fast, ratio)
    f2s = make_f2s(a, nx_fast, nx_slow, dx_fast, ratio)
    s2s = make_s2s(a, nx_slow, dx_fast, ratio)
    # Build and run the method.
    method = make_multirate_method(f2f, s2f, f2s, s2s, ratio)
    space, init = make_multirate_initial_conditions(dx_fast, ratio)
    y_slow = init[0:nx_slow]
    y_fast = init[nx_slow:]
    history = run_multirate_method(method, y_fast, y_slow, dt, t_start, t_end)
    # Graph the resulting lines.
    time_steps = int((t_end - t_start) / dt)
    xs, ys = np.meshgrid(space, np.linspace(t_start, t_end, time_steps))
    zs = np.array(history[0:time_steps])
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot_wireframe(xs, ys, zs, rstride=len(ys), cstride=1)
    plt.show()


if __name__ == '__main__':
    # plot_multirate_initial_conditions(dx=0.02, ratio=5)
    plot_multirate_history(a=1.5, dx_fast=0.02, ratio=5, dt=0.001, t_start=0,
                           t_end=0.5)
