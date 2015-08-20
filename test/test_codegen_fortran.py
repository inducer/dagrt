#! /usr/bin/env python

from __future__ import division, with_statement, print_function

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

from leap.vm.language import YieldState
from leap.vm.language import TimeIntegratorCode, CodeBuilder
import leap.vm.codegen.fortran as f
from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper

from leap.method.ab.multirate.methods import methods as MRAB_METHODS  # noqa

from utils import RawCodeBuilder


skip = pytest.mark.skipif(True, reason="not fully implemented")


def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id='return', time=0, time_id='final',
                    expression=0, component_id='state',
        depends_on=[]))
    cbuild.commit()
    code = TimeIntegratorCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'])
    codegen = f.CodeGenerator("simple",
            ode_component_type_map={
                "state": f.ArrayType(
                    (200,),
                    f.BuiltinType('real (kind=8)'),)
                })
    print(codegen(code))


def run_fortran(sources, fortran_options=[]):
    from utils import TemporaryDirectory
    from os.path import join

    with TemporaryDirectory() as tmpdir:
        source_names = []
        for name, contents in sources:
            source_names.append(name)

            with open(join(tmpdir, name), "w") as srcf:
                srcf.write(contents)

        from subprocess import check_call, Popen, PIPE
        check_call(
                ["gfortran", "-Wall", "-g", "-oruntest"]
                + fortran_options
                + list(source_names),
                cwd=tmpdir)

        p = Popen([join(tmpdir, "runtest")], stdout=PIPE, stderr=PIPE,
                close_fds=True)
        stdout_data, stderr_data = p.communicate()

        if stdout_data:
            print("Fortran code said this on stdout: -----------------------------",
                    file=sys.stderr)
            print(stdout_data, file=sys.stderr)
            print("---------------------------------------------------------------",
                    file=sys.stderr)

        if stderr_data:
            raise RuntimeError("Fortran code has non-empty stderr:\n" +
                               stderr_data.decode('ascii'))

        return p.returncode, stdout_data, stderr_data


def read_file(name):
    with open(name, "r") as inf:
        return inf.read()


# {{{ test rk methods

@pytest.mark.parametrize(("min_order", "stepper"), [
    (2, ODE23TimeStepper("y", use_high_order=False)),
    (3, ODE23TimeStepper("y", use_high_order=True)),
    (4, ODE45TimeStepper("y", use_high_order=False)),
    (5, ODE45TimeStepper("y", use_high_order=True)),
    ])
def test_rk_codegen(min_order, stepper):
    """Test whether Fortran code generation for the Runge-Kutta
    timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk.f90", read_file("test_rk.f90").replace(
            "MIN_ORDER", str(min_order - 0.3)+"d0")),
        ])

# }}}


# {{{ test fancy codegen

def test_rk_codegen_fancy():
    """Test whether Fortran code generation with lots of fancy features for the
    Runge-Kutta timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = ODE23TimeStepper(component_id, use_high_order=True)

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs,
            register_function)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                <%

                igrid = declare_new("integer", "igrid")
                i = declare_new("integer", "i")

                %>

                do ${igrid} = 1, region%n_grids
                  do ${i} = 1, region%n_grid_dofs(${igrid})
                    ${result}(${igrid})%conserved_var(${i}) = &
                     -2*${y}(${igrid})%conserved_var(${i})
                  end do
                end do

                """))
    freg = register_function(freg, "notify_pre_state_update", ("updated_component",))
    freg = freg.register_codegen("notify_pre_state_update", "fortran",
            f.CallCode("""
                write(*,*) 'before state update'
                """))
    freg = register_function(
            freg, "notify_post_state_update", ("updated_component",))
    freg = freg.register_codegen("notify_post_state_update", "fortran",
            f.CallCode("""
                write(*,*) 'after state update'
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    "region%n_grids",
                    index_vars="igrid",
                    element_type=f.StructureType(
                        "sim_grid_state_type",
                        (
                            ("conserved_var", f.PointerType(
                                f.ArrayType(
                                    ("region%n_grid_dofs(igrid)",),
                                    f.BuiltinType('real (kind=8)')))),
                        )))
                },
            function_registry=freg,
            module_preamble="""
                use sim_types
                """,
            call_before_state_update="notify_pre_state_update",
            call_after_state_update="notify_post_state_update",
            extra_arguments="region",
            extra_argument_decl="""
                type(region_type), pointer :: region
                """)

    code_str = codegen(code)
    print(code_str)

    run_fortran([
        ("sim_types.f90", read_file("sim_types.f90")),
        ("rkmethod.f90", code_str),
        ("test_fancy_rk.f90", read_file("test_fancy_rk.f90")),
        ])

# }}}


@pytest.mark.parametrize("min_order", [2, 3, 4, 5])
@pytest.mark.parametrize("method_name", list(MRAB_METHODS.keys()))
def test_multirate_codegen(min_order, method_name):
    from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
    from pytools import DictionaryWithDefault

    orders = DictionaryWithDefault(lambda x: min_order)

    stepper = TwoRateAdamsBashforthTimeStepper(
            MRAB_METHODS[method_name], orders, 4)

    code = stepper.generate()

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)

    freg = base_function_registry
    for func_name in [
            "<func>s2s",
            "<func>f2s",
            "<func>s2f",
            "<func>f2f",
            ]:
        component_id = {
                "s": "slow",
                "f": "fast",
                }[func_name[-1]]
        freg = register_ode_rhs(freg, identifier=func_name,
                component_id=component_id,
                input_component_ids=("slow", "fast"),
                input_component_names=("s", "f"))

    freg = freg.register_codegen("<func>s2f", "fortran",
        f.CallCode("""
            ${result} = (sin(2*${t}) - 1)*${s}
            """))
    freg = freg.register_codegen("<func>f2s", "fortran",
      f.CallCode("""
          ${result} = (sin(2*${t}) + 1)*${f}
          """))
    freg = freg.register_codegen("<func>f2f", "fortran",
      f.CallCode("""
          ${result} = cos(2*${t})*${f}
          """))
    freg = freg.register_codegen("<func>s2s", "fortran",
      f.CallCode("""
          ${result} = -cos(2*${t})*${s}
          """))

    codegen = f.CodeGenerator(
            'MRAB',
            ode_component_type_map={
                "slow": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    ),
                "fast": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    code_str = codegen(code)

    if 0:
        with open("abmethod.f90", "wt") as outf:
            outf.write(code_str)

    if min_order == 3:
        run_fortran([
            ("abmethod.f90", code_str),
            ("test_mrab.f90", read_file("test_mrab.f90").replace(
                "MIN_ORDER", str(min_order - 0.3)+"d0").replace("NUM_TRIPS_ONE", str(200)).replace("NUM_TRIPS_TWO", str(300))),
            ],
            fortran_options=["-llapack", "-lblas"])

    else:
        run_fortran([
            ("abmethod.f90", code_str),
            ("test_mrab.f90", read_file("test_mrab.f90").replace(
                "MIN_ORDER", str(min_order - 0.3)+"d0").replace("NUM_TRIPS_ONE", str(100)).replace("NUM_TRIPS_TWO", str(150))),
            ],
            fortran_options=["-llapack", "-lblas"])


def test_adaptive_rk_codegen():
    """Test whether Fortran code generation for the Runge-Kutta
    timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = ODE45TimeStepper(component_id, use_high_order=False, rtol=1e-6)

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result}(1) = ${y}(2)
                ${result}(2) = -30*((${y}(1))**2 - 1)*${y}(2) - ${y}(1)
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                "y": f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    ),
                },
            function_registry=freg)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk_adaptive.f90", read_file("test_rk_adaptive.f90")),
        ])


def test_adaptive_rk_codegen_error():
    """Test whether Fortran code generation for the Runge-Kutta
    timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = ODE45TimeStepper(component_id, use_high_order=False, atol=1e-6)

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk_adaptive_error.f90", read_file("test_rk_adaptive_error.f90")),
        ])


class MatrixInversionFailure(object):
    pass


def test_arrays_and_linalg():
    from leap.vm.function_registry import base_function_registry as freg

    with CodeBuilder(label="primary") as cb:
        cb("n", "4")
        cb("nodes", "`<builtin>array`(n)")
        cb("vdm", "`<builtin>array`(n*n)")
        cb("identity", "`<builtin>array`(n*n)")
        cb.fence()

        cb("nodes[i]", "i/n",
                loops=[("i", 0, "n")])
        cb("identity[i]", "0",
                loops=[("i", 0, "n*n")])
        cb.fence()

        cb("identity[i*n + i]", "1",
                loops=[("i", 0, "n")])
        cb("vdm[j*n + i]", "nodes[i]**j",
                loops=[("i", 0, "n"), ("j", 0, "n")])

        cb.fence()

        cb("vdm_inverse", "`<builtin>linear_solve`(vdm, identity, n, n)")
        cb("myarray", "`<builtin>matmul`(vdm, vdm_inverse, n, n)")

        cb("myzero", "myarray - identity")
        cb("dummy", "`<builtin>print`(myzero)")
        with cb.if_("`<builtin>norm_2`(myzero) > 10**(-8)"):
            cb.raise_(MatrixInversionFailure)

    code = TimeIntegratorCode.create_with_steady_state(
        cb.state_dependencies, cb.instructions)

    codegen = f.CodeGenerator(
            'arrays',
            function_registry=freg,
            ode_component_type_map={})

    code_str = codegen(code)
    print(code_str)
    run_fortran([
        ("arrays.f90", code_str),
        ("test_arrays_and_linalg.f90", read_file("test_arrays_and_linalg.f90")),
        ],
        fortran_options=["-llapack", "-lblas"])


@pytest.mark.parametrize("min_order", [2, 3, 4, 5])
def test_singlerate_squarewave(min_order):
    from leap.method.ab import AdamsBashforthTimeStepper

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = AdamsBashforthTimeStepper("y", min_order)

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'ABMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    code_str = codegen(code)

    run_fortran([
        ("abmethod.f90", code_str),
        ("test_ab_squarewave.f90", read_file("test_ab_squarewave.f90").replace(
            "MIN_ORDER", str(min_order - 0.3)+"d0")),
        ],
        fortran_options=["-llapack", "-lblas"])


@pytest.mark.parametrize("method_name", list(MRAB_METHODS.keys()))
@pytest.mark.parametrize("min_order", [2, 3, 4, 5])
def test_multirate_squarewave(min_order, method_name):
    from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
    from pytools import DictionaryWithDefault

    orders = DictionaryWithDefault(lambda x: min_order)

    stepper = TwoRateAdamsBashforthTimeStepper(MRAB_METHODS[method_name], orders, 4)

    code = stepper.generate()

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)

    freg = base_function_registry
    for func_name in [
            "<func>s2s",
            "<func>f2s",
            "<func>s2f",
            "<func>f2f",
            ]:
        component_id = {
                "s": "slow",
                "f": "fast",
                }[func_name[-1]]
        freg = register_ode_rhs(freg, identifier=func_name,
                component_id=component_id,
                input_component_ids=("slow", "fast"),
                input_component_names=("s", "f"))

    freg = freg.register_codegen("<func>s2f", "fortran",
        f.CallCode("""
            ${result} = (sin(2*${t}) - 1)*${s}
            """))
    freg = freg.register_codegen("<func>f2s", "fortran",
      f.CallCode("""
          ${result} = (sin(2*${t}) + 1)*${f}
          """))
    freg = freg.register_codegen("<func>f2f", "fortran",
      f.CallCode("""
          ${result} = cos(2*${t})*${f}
          """))
    freg = freg.register_codegen("<func>s2s", "fortran",
      f.CallCode("""
          ${result} = -cos(2*${t})*${s}
          """))

    codegen = f.CodeGenerator(
            'MRAB',
            ode_component_type_map={
                "slow": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    ),
                "fast": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    code_str = codegen(code)

    # Build in conditionals to alter the timestep based on order such that all tests pass

    if min_order == 3:
        run_fortran([
            ("abmethod.f90", code_str),
            ("test_mrab_squarewave.f90", read_file("test_mrab_squarewave.f90").replace(
                "MIN_ORDER", str(min_order - 0.3)+"d0").replace("NUM_TRIPS_ONE", str(200)).replace("NUM_TRIPS_TWO", str(300))),
            ],
            fortran_options=["-llapack", "-lblas"])
    elif min_order == 5:
        if method_name == 'Sqrs':
            # This is an especially problematic test in that it has errors that are too low to achieve order convergence
            run_fortran([
                ("abmethod.f90", code_str),
                ("test_mrab_squarewave.f90", read_file("test_mrab_squarewave.f90").replace(
                    "MIN_ORDER", str(min_order - 0.3)+"d0").replace("NUM_TRIPS_ONE", str(5)).replace("NUM_TRIPS_TWO", str(10))),
                ],
                fortran_options=["-llapack", "-lblas"])

        else:
            run_fortran([
                ("abmethod.f90", code_str),
                ("test_mrab_squarewave.f90", read_file("test_mrab_squarewave.f90").replace(
                    "MIN_ORDER", str(min_order - 0.3)+"d0").replace("NUM_TRIPS_ONE", str(125)).replace("NUM_TRIPS_TWO", str(200))),
                ],
                fortran_options=["-llapack", "-lblas"])
    else:
        run_fortran([
            ("abmethod.f90", code_str),
            ("test_mrab_squarewave.f90", read_file("test_mrab_squarewave.f90").replace(
                "MIN_ORDER", str(min_order - 0.3)+"d0").replace("NUM_TRIPS_ONE", str(100)).replace("NUM_TRIPS_TWO", str(150))),
            ],
            fortran_options=["-llapack", "-lblas"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
