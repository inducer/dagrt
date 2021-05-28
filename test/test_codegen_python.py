#! /usr/bin/env python

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

import sys
import pytest

import numpy.linalg as la
import numpy as np

from dagrt.language import Assign, YieldState, FailStep, Raise, Nop
from dagrt.language import CodeBuilder, DAGCode
from dagrt.codegen import PythonCodeGenerator
from pymbolic import var

from utils import (  # noqa
        RawCodeBuilder, python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg,
        create_DAGCode_with_init_and_main_phases,
        create_DAGCode_with_steady_phase)


def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id="return", time=0, time_id="final",
                    expression=0, component_id="<state>",
        depends_on=[]))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    print(codegen(code))
    Method = codegen.get_class(code)  # noqa
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    hist = [s for s in method.run(max_steps=2)]
    assert len(hist) == 3
    assert isinstance(hist[0], method.StepCompleted)
    assert hist[0].current_phase == "init"
    assert isinstance(hist[1], method.StateComputed)
    assert hist[1].state_component == 0
    assert isinstance(hist[2], method.StepCompleted)
    assert hist[2].current_phase == "main"


def test_basic_conditional_codegen():
    """Test whether the code generator generates branches properly."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        Assign(
            id="then_branch",
            assignee="<state>y", assignee_subscript=(),
            expression=1, condition=True),
        Assign(id="else_branch",
            assignee="<state>y", assignee_subscript=(),
            expression=0, condition=False),
        Nop(id="branch", depends_on=["then_branch", "else_branch"]),
        YieldState(id="return", time=0, time_id="final",
            expression=var("<state>y"), component_id="<state>",
        depends_on=["branch"]))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={"y": 6})
    hist = [s for s in method.run(max_steps=2)]
    assert len(hist) == 3
    assert isinstance(hist[1], method.StateComputed)
    assert hist[1].state_component == 1
    assert isinstance(hist[2], method.StepCompleted)


def test_basic_assign_rhs_codegen():
    """Test whether the code generator generates RHS evaluation code
    properly."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        Assign(id="assign_rhs1",
                         assignee="<state>y",
                         assignee_subscript=(),
                         expression=var("y")(t=var("<t>")),
                         depends_on=[]),
        Assign(id="assign_rhs2",
                         assignee="<state>y",
                         assignee_subscript=(),
                         expression=var("yy")(t=var("<t>"), y=var("<state>y")),
                         depends_on=["assign_rhs1"]),
        YieldState(id="return", time=0, time_id="final",
            expression=var("<state>y"), component_id="<state>",
            depends_on=["assign_rhs2"])
        )
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa

    def y(t):
        return 6

    def yy(t, y):
        return y + 6

    method = Method({"y": y, "yy": yy})
    method.set_up(t_start=0, dt_start=0, context={"y": 0})
    hist = [s for s in method.run(max_steps=2)]
    assert len(hist) == 3
    assert isinstance(hist[1], method.StateComputed)
    assert hist[1].state_component == 12
    assert isinstance(hist[2], method.StepCompleted)


def test_basic_raise_codegen():
    """Test code generation of the Raise statement."""
    cbuild = RawCodeBuilder()

    class TimeStepUnderflow(RuntimeError):
        pass

    cbuild.add_and_get_ids(Raise(TimeStepUnderflow, "underflow", id="raise"))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    try:
        # initialization
        for _result in method.run_single_step():
            pass
        # first primary step
        for _result in method.run_single_step():
            raise AssertionError()
    except method.TimeStepUnderflow:
        pass
    except Method.StepError as e:
        assert e.condition == "TimeStepUnderflow"
    except Exception as e:
        assert not e, e


def test_basic_fail_step_codegen():
    """Test code generation of the FailStep statement."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(FailStep(id="fail"))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    print(codegen(code))

    for _evt in method.run_single_step():
        pass

    with pytest.raises(method.FailStepException):
        for evt in method.run_single_step():
            print(evt)


def test_local_name_distinctness():
    """Test whether the code generator gives locals distinct names."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        Assign(
            id="assign_y^",
            assignee="y^", assignee_subscript=(), expression=1),
        Assign(
            id="assign_y*",
            assignee="y*", assignee_subscript=(), expression=0),
        YieldState(id="return", time=0, time_id="final",
            expression=var("y^") + var("y*"),
            component_id="y", depends_on=["assign_y^", "assign_y*"]))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    hist = list(method.run(max_steps=2))
    assert len(hist) == 3
    assert isinstance(hist[1], method.StateComputed)
    assert hist[1].state_component == 1


def test_global_name_distinctness():
    """Test whether the code generator gives globals distinct names."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        Assign(
            id="assign_y^",
            assignee="<p>y^", assignee_subscript=(), expression=1),
        Assign(
            id="assign_y*",
            assignee="<p>y*", assignee_subscript=(), expression=0),
        YieldState(id="return", time=0, time_id="final",
            expression=var("<p>y^") + var("<p>y*"),
            component_id="y", depends_on=["assign_y^", "assign_y*"]))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa
    method = Method({})
    method.set_up(t_start=0, dt_start=0, context={})
    hist = list(method.run(max_steps=2))
    assert len(hist) == 3
    assert isinstance(hist[1], method.StateComputed)
    assert hist[1].state_component == 1


def test_function_name_distinctness():
    """Test whether the code generator gives functions distinct names."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id="return", time=0, time_id="final",
            expression=var("<func>y^")() + var("<func>y*")(),
            component_id="y"))
    cbuild.commit()
    code = create_DAGCode_with_init_and_main_phases(
            init_statements=[],
            main_statements=cbuild.statements)
    codegen = PythonCodeGenerator(class_name="Method")
    Method = codegen.get_class(code)  # noqa
    method = Method({"<func>y^": lambda: 0,
                     "<func>y*": lambda: 1})
    method.set_up(t_start=0, dt_start=0, context={})
    hist = list(method.run(max_steps=2))
    assert len(hist) == 3
    assert isinstance(hist[1], method.StateComputed)
    assert hist[1].state_component == 1


def test_switch_phases(python_method_impl):
    from dagrt.language import CodeBuilder, ExecutionPhase

    with CodeBuilder(name="state_1") as builder_1:
        builder_1(var("<state>x"), 1)
        builder_1.switch_phase("state_2")
    with CodeBuilder(name="state_2") as builder_2:
        builder_2.yield_state(var("<state>x"), "x", 0, "final")

    code = DAGCode(
        phases={
            "state_1": ExecutionPhase(
                name="state_1",
                next_phase="state_1",
                statements=builder_1.statements),
            "state_2": ExecutionPhase(
                name="state_2",
                next_phase="state_2",
                statements=builder_2.statements)
        },
        initial_phase="state_1")
    from utils import execute_and_return_single_result
    result = execute_and_return_single_result(python_method_impl, code,
                                              initial_context={"x": 0},
                                              max_steps=2)
    assert result == 1


def get_IfThenElse_test_code_and_expected_result():
    from dagrt.expression import IfThenElse

    with CodeBuilder(name="primary") as cb:
        cb(var("c1"), IfThenElse(True, 0, 1))
        cb(var("c2"), IfThenElse(False, 0, 1))
        cb(var("c3"), IfThenElse(IfThenElse(True, True, False), 0, 1))
        cb(var("c4"), IfThenElse(IfThenElse(False, True, False), 0, 1))
        cb(var("c5"), IfThenElse(True, IfThenElse(True, 0, 1), 2))
        cb(var("c6"), IfThenElse(True, IfThenElse(False, 0, 1), 2))
        cb(var("c7"), IfThenElse(False, 0, IfThenElse(True, 1, 2)))
        cb(var("c8"), IfThenElse(False, 0, IfThenElse(False, 1, 2)))
        cb(var("c9"), 1 + IfThenElse(True, 0, 1))
        cb(var("c10"), 1 + IfThenElse(False, 0, 1))
        cb.yield_state(tuple(var("c" + str(i)) for i in range(1, 11)),
                       "result", 0, "final")

    code = create_DAGCode_with_steady_phase(cb.statements)

    return (code, (0, 1, 0, 1, 0, 1, 1, 2, 1, 2))


def test_IfThenElse(python_method_impl):
    from utils import execute_and_return_single_result
    code, expected_result = get_IfThenElse_test_code_and_expected_result()
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == expected_result


def test_arrays_and_looping(python_method_impl):
    with CodeBuilder(name="primary") as cb:
        cb("myarray", "`<builtin>array`(20)")
        cb("myarray[i]", "i", loops=[("i", 0, 20)])
        cb.yield_state("myarray[15]", "result", 0, "final")

    from utils import execute_and_return_single_result

    code = create_DAGCode_with_steady_phase(cb.statements)
    result = execute_and_return_single_result(python_method_impl, code)
    assert result == 15


def test_arrays_and_linalg(python_method_impl):
    with CodeBuilder(name="primary") as cb:
        cb("n", "4")
        cb("nodes", "`<builtin>array`(n)")
        cb("vdm", "`<builtin>array`(n*n)")
        cb("identity", "`<builtin>array`(n*n)")

        cb("nodes[i]", "i/n",
                loops=[("i", 0, "n")])
        cb("identity[i]", "0",
                loops=[("i", 0, "n*n")])

        cb("identity[i*n + i]", "1",
                loops=[("i", 0, "n")])
        cb("vdm[j*n + i]", "nodes[i]**j",
                loops=[("i", 0, "n"), ("j", 0, "n")])

        cb("vdm_inverse", "`<builtin>linear_solve`(vdm, identity, n, n)")
        cb("myarray", "`<builtin>matmul`(vdm, vdm_inverse, n, n)")

        cb((), "`<builtin>print`(myarray)")

        cb.yield_state("myarray", "result", 0, "final")

    from utils import execute_and_return_single_result

    code = create_DAGCode_with_steady_phase(cb.statements)
    result = execute_and_return_single_result(python_method_impl, code)

    result = result.reshape(4, 4, order="F")

    assert la.norm(result - np.eye(4)) < 1e-10


def test_svd(python_method_impl):
    with CodeBuilder(name="primary") as cb:
        cb("n", 3)
        cb("nodes", "`<builtin>array`(n)")
        cb("vdm", "`<builtin>array`(n*n)")
        cb("identity", "`<builtin>array`(n*n)")

        cb("nodes[i]", "i/n",
                loops=[("i", 0, "n")])

        cb("vdm[j*n + i]", "nodes[i]**j",
                loops=[("i", 0, "n"), ("j", 0, "n")])

        cb("vdm_u, vdm_sigma, vdm_vt", "`<builtin>svd`(vdm, n)")
        cb("vdm_usigma", "`<builtin>array`(n*n)")
        cb("vdm_v", "`<builtin>array`(n*n)")
        cb("vdm_usigma[i + j*n]", "vdm_u[i + j*n] * vdm_sigma[j]",
                loops=[("i", 0, "n"), ("j", 0, "n")])
        cb("vdm_v[i + j*n]", "vdm_vt[j + i*n]",
                loops=[("i", 0, "n"), ("j", 0, "n")])

        cb("vdm_2", "`<builtin>matmul`(vdm_usigma, vdm_vt, n, n)")
        cb("diff", "vdm-vdm_2")

        cb((), "`<builtin>print`(diff)")

        cb.yield_state("diff", "result", 0, "final")

    from utils import execute_and_return_single_result

    code = create_DAGCode_with_steady_phase(cb.statements)
    result = execute_and_return_single_result(python_method_impl, code)

    assert la.norm(result) < 1e-10


def test_class_preamble():
    from dagrt.language import CodeBuilder

    with CodeBuilder(name="primary") as cb:
        cb.assign("<t>", "<t> + <dt>")
        cb.yield_state("f()", "f", 0, "final")

    code = create_DAGCode_with_steady_phase(cb.statements)

    from dagrt.codegen import PythonCodeGenerator
    import dagrt.function_registry as freg

    preamble = """
            @staticmethod
            def f():
                return 1
    """

    f = freg.Function(
            identifier="f",
            language_to_codegen={"python": lambda *args: "self.f()"})

    generator = PythonCodeGenerator(
            "PythonMethod",
            class_preamble=preamble,
            function_registry=freg.base_function_registry.register(f))

    class_ = generator.get_class(code)

    method = class_(function_map={})
    method.set_up(t_start=0, dt_start=1, context={})

    events = list(method.run(t_end=1))
    assert events
    assert isinstance(events[0], class_.StateComputed)
    assert events[0].state_component == 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
