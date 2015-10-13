"""Various usefulness"""
from __future__ import division, with_statement

import numpy as np


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



# {{{ things to pass for python_method_impl

def python_method_impl_interpreter(code, **kwargs):
    from dagrt.exec_numpy import NumpyInterpreter
    return NumpyInterpreter(code, **kwargs)


def python_method_impl_codegen(code, **kwargs):
    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name='Method')
    return codegen.get_class(code)(**kwargs)

# }}}


def execute_and_return_single_result(python_method_impl, code, initial_context={},
                                     max_steps=1):
    interpreter = python_method_impl(code, function_map={})
    interpreter.set_up(t_start=0, dt_start=0, context=initial_context)
    has_state_component = False
    for event in interpreter.run(max_steps=max_steps):
        if isinstance(event, interpreter.StateComputed):
            has_state_component = True
            state_component = event.state_component
    assert has_state_component
    return state_component


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
                             problem=DefaultProblem(), dts=_default_dts,
                             show_dag=False, plot_solution=False):
    component_id = method.component_id
    code = method.generate()
    print(code)

    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for dt in dts:
        t = problem.t_start
        y = problem.initial()
        final_t = problem.t_end

        interp = method_impl(code, function_map={
            "<func>" + component_id: problem,
            })
        interp.set_up(t_start=t, dt_start=dt, context={component_id: y})

        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, interp.StateComputed):
                assert event.component_id == component_id
                values.append(event.state_component[0])
                times.append(event.t)

        assert abs(times[-1] - final_t) / final_t < 0.1

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


# {{{ temporary directory

class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    # Yanked from
    # https://hg.python.org/cpython/file/3.3/Lib/tempfile.py

    # Handle mkdtemp raising an exception
    name = None
    _closed = False

    def __init__(self, suffix="", prefix="tmp", dir=None):
        from tempfile import mkdtemp
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        import warnings
        if self.name and not self._closed:
            from shutil import rmtree
            try:
                rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                if "None" not in '%s' % (ex,):
                    raise
                self._rmtree(self.name)
            self._closed = True
            if _warn and warnings.warn:
                warnings.warn("Implicitly cleaning up {!r}".format(self))

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

# }}}


# {{{ low-level code building utility

class RawCodeBuilder(object):

    def __init__(self):
        self.id_set = set()
        self.generated_id_set = set()

        self._instructions = []
        self.build_group = []

    def fresh_insn_id(self, prefix):
        """Return an instruction name that is guaranteed not to be in use and
        not to be generated in the future."""
        from pytools import generate_unique_names
        for possible_id in generate_unique_names(prefix):
            if possible_id not in self.id_set and possible_id not in \
                    self.generated_id_set:
                self.generated_id_set.add(possible_id)
                return possible_id

    def add_and_get_ids(self, *insns):
        new_ids = []
        for insn in insns:
            set_attrs = {}
            if not hasattr(insn, "id") or insn.id is None:
                set_attrs["id"] = self.fresh_insn_id("insn")
            else:
                if insn.id in self.id_set:
                    raise ValueError("duplicate ID")

            if not hasattr(insn, "depends_on"):
                set_attrs["depends_on"] = frozenset()

            if set_attrs:
                insn = insn.copy(**set_attrs)

            self.build_group.append(insn)
            new_ids.append(insn.id)

        # For exception safety, only make state change at end.
        self.id_set.update(new_ids)
        return new_ids

    def commit(self):
        for insn in self.build_group:
            for dep in insn.depends_on:
                if dep not in self.id_set:
                    raise ValueError("unknown dependency id: %s" % dep)

        self._instructions.extend(self.build_group)
        del self.build_group[:]

    @property
    def instructions(self):
        if self.build_group:
            raise ValueError("attempted to get instructions while "
                    "build group is uncommitted")

        return self._instructions

# }}}

# vim: foldmethod=marker
