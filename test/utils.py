"""Various usefulness"""

from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner, Matt Wala"

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


class Problem(object):
    """
    .. attribute :: t_start
    .. attribute :: t_end
    """

    def initial(self):
        """Return an initial value."""
        raise NotImplementedError()

    def exact(self, t):
        """Return the exact solution, if available."""
        raise NotImplementedError()

    def __call__(self, t, y):
        raise NotImplementedError()


class DefaultProblem(Problem):

    t_start = 1

    t_end = 10

    def initial(self):
        return np.array([1, 3], dtype=np.float64)

    def exact(self, t):
        inner = np.sqrt(3) / 2 * np.log(t)
        return np.sqrt(t) * (
                5 * np.sqrt(3) / 3 * np.sin(inner)
                + np.cos(inner)
                )

    def __call__(self, t, y):
        u, v = y
        return np.array([v, -u / t ** 2], dtype=np.float64)


_default_dts = 2 ** -np.array(range(4, 7), dtype=np.float64)


def check_simple_convergence(method, method_impl, expected_order,
                             problem=DefaultProblem(),
                             dts=_default_dts, show_dag=False,
                             plot_solution=False):
    component_id = "y"
    code = method(component_id)

    if show_dag:
        from leap.vm.language import show_dependency_graph
        show_dependency_graph(code)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for dt in dts:
        t = problem.t_start
        y = problem.initial()
        final_t = problem.t_end

        interp = method_impl(code, function_map={component_id: problem})
        interp.set_up(t_start=t, dt_start=dt, state={component_id: y})
        interp.initialize()

        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, interp.StateComputed):
                assert event.component_id == component_id
                values.append(event.state_component[0])
                times.append(event.t)

        assert abs(times[-1] - final_t) < 1e-10

        times = np.array(times)

        if plot_solution:
            import matplotlib.pyplot as pt
            pt.plot(times, values, label="comp")
            pt.plot(times, problem.exact(times), label="true")
            pt.show()

        error = abs(values[-1] - problem.exact(final_t))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("%s: expected order %d" % (method.__class__.__name__,
                                     expected_order))
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > expected_order * 0.9
