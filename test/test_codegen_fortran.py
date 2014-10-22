#! /usr/bin/env python

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

import sys
import pytest

from leap.vm.language import AssignExpression, AssignNorm, AssignRHS, If, \
    ReturnState
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
        ReturnState(id='return', time=0, time_id='final',
                    expression=0, component_id='<state>',
        depends_on=[]))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = FortranCodeGenerator("simple")
    print(codegen(code))


@pytest.mark.parametrize("stepper", [
    ODE23TimeStepper(use_high_order=False),
    ODE23TimeStepper(use_high_order=True),
    ODE45TimeStepper(use_high_order=False),
    ODE45TimeStepper(use_high_order=True),
    ])
def test_rk_codegen(stepper):
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
            'RKMethod', freg,
            ode_component_type_map={
                component_id: FortranType('real (kind=8)', (200,))
                },
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    print(codegen(code))


@skip  # FIXME remove when done
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
        component_id = func_name[-1]
        freg = register_ode_rhs(freg, identifier=func_name,
                component_id=component_id,
                input_component_ids=("s", "f"))
        freg = freg.register_codegen(func_name, "fortran",
                FortranCallCode("""
                    ${assignee} = -2*${f} + ${s}
                    """))

    codegen = FortranCodeGenerator(
            'RKMethod', freg,
            ode_component_type_map={
                "s": FortranType('real (kind=8)', (200,), ),
                "f": FortranType('real (kind=8)', (300,), )
                },
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
