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
from leap.vm.language import CodeBuilder, TimeIntegratorCode
from leap.vm.codegen.fortran import (
        FortranCodeGenerator, FortranType, FortranCallCode)
from leap.method.rk import ODE23TimeStepper, ODE45TimeStepper


skip = pytest.mark.skipif(True, reason="not fully implemented")


def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id='return', time=0, time_id='final',
                    expression=0, component_id='state',
        depends_on=[]))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = FortranCodeGenerator("simple",
            ode_component_type_map={
                "state": FortranType('real (kind=8)', (200,))
                })
    print(codegen(code))


def run_fortran(sources):
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
                ["gfortran", "-g", "-oruntest"] + list(source_names),
                cwd=tmpdir)

        p = Popen([join(tmpdir, "runtest")], stdout=PIPE, stderr=PIPE,
                close_fds=True)
        stdout_data, stderr_data = p.communicate()

        if stderr_data:
            raise RuntimeError("Fortran code has non-empty stderr:\n"+stderr_data)

        return p.returncode, stdout_data, stderr_data


# {{{ test rk methods

TEST_RK_F90 = """
program test_rkmethod
  use RKMethod, only: leap_state_type, &
    timestep_initialize => initialize, &
    timestep_shutdown => shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  implicit none

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition, true_sol
  integer, dimension(2) :: nsteps

  integer run_count
  real*8 t_fin
  parameter (run_count=2, t_fin=1d0)

  real*8, dimension(run_count):: dt_values, errors

  real*8 est_order, min_order

  integer stderr
  parameter(stderr=0)

  integer istep, irun

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 2
  initial_condition(2) = 2.3
  true_sol = initial_condition * exp(-2*t_fin)

  nsteps(1) = 20
  nsteps(2) = 50

  do irun = 1,run_count
    dt_values(irun) = t_fin/nsteps(irun)

    call timestep_initialize( &
      leap_state=state_ptr, &
      state_y=initial_condition, &
      leap_t=0d0, &
      leap_dt=dt_values(irun))

    call leap_state_func_initialization(leap_state=state_ptr)
    do istep = 1,nsteps(irun)
      call leap_state_func_primary(leap_state=state_ptr)
      write(*,*) state%ret_state_y
    enddo

    errors(irun) = sqrt(sum((true_sol-state%ret_state_y)**2))

    write(*,*) errors

    call timestep_shutdown(leap_state=state_ptr)
    write(*,*) 'done'
  enddo

  min_order = MIN_ORDER
  est_order = log(errors(2)/errors(1))/log(dt_values(2)/dt_values(1))

  write(*,*) 'estimated order:', est_order
  if (est_order < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order, ' < ', &
        min_order
  endif

end program
"""


@pytest.mark.parametrize(("min_order", "stepper"), [
    (2, ODE23TimeStepper(use_high_order=False)),
    (3, ODE23TimeStepper(use_high_order=True)),
    (4, ODE45TimeStepper(use_high_order=False)),
    (5, ODE45TimeStepper(use_high_order=True)),
    ])
def test_rk_codegen(min_order, stepper):
    """Test whether Runge-Kutta timestepper code generation works."""

    component_id = 'y'

    from leap.vm.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id)
    freg = freg.register_codegen(component_id, "fortran",
            FortranCallCode("""
                ${assignee} = -2*${y}
                """))

    code = stepper(component_id)

    codegen = FortranCodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: FortranType('real (kind=8)', (2,))
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk.f90", TEST_RK_F90.replace("MIN_ORDER", str(min_order - 0.3)+"d0")),
        ])

# }}}


def test_multirate_codegen():
    from leap.method.ab.multirate import TwoRateAdamsBashforthTimeStepper
    from leap.method.ab.multirate.methods import methods
    from pytools import DictionaryWithDefault

    order = DictionaryWithDefault(lambda x: 4)

    stepper = TwoRateAdamsBashforthTimeStepper(methods['F'], order, 4)

    code = stepper()

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
        freg = freg.register_codegen(func_name, "fortran",
                FortranCallCode("""
                    ${assignee} = -2*${f} + ${s}
                    """))

    codegen = FortranCodeGenerator(
            'RKMethod',
            ode_component_type_map={
                "slow": FortranType('real (kind=8)', (200,), ),
                "fast": FortranType('real (kind=8)', (300,), )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    print(codegen(code))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
