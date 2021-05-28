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

from dagrt.language import YieldState
from dagrt.language import CodeBuilder
import dagrt.codegen.fortran as f

from utils import RawCodeBuilder

from dagrt.utils import run_fortran

from utils import create_DAGCode_with_steady_phase

#skip = pytest.mark.skipif(True, reason="not fully implemented")


def read_file(rel_path):
    from os.path import join, abspath, dirname
    path = join(abspath(dirname(__file__)), rel_path)
    with open(path) as inf:
        return inf.read()


def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id="return", time=0, time_id="final",
                    expression=0, component_id="state",
        depends_on=[]))
    cbuild.commit()
    code = create_DAGCode_with_steady_phase(cbuild.statements)
    codegen = f.CodeGenerator("simple",
            user_type_map={
                "state": f.ArrayType(
                    (200,),
                    f.BuiltinType("real (kind=8)"),)
                })
    print(codegen(code))


class MatrixInversionFailure:
    pass


def test_arrays_and_linalg():
    from dagrt.function_registry import base_function_registry as freg

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

        cb("myzero", "myarray - identity")
        cb((), "`<builtin>print`(myzero)")
        with cb.if_("`<builtin>norm_2`(myzero) > 10**(-8)"):
            cb.raise_(MatrixInversionFailure)

    code = create_DAGCode_with_steady_phase(cb.statements)

    codegen = f.CodeGenerator(
            "arrays",
            function_registry=freg,
            user_type_map={},
            emit_instrumentation=True,
            timing_function="second")

    code_str = codegen(code)
    if 0:
        with open("arrays.f90", "wt") as outf:
            outf.write(code_str)

    run_fortran([
        ("arrays.f90", code_str),
        ("test_arrays_and_linalg.f90", read_file("test_arrays_and_linalg.f90")),
        ],
        fortran_libraries=["lapack", "blas"])


def test_self_dep_in_loop():
    with CodeBuilder(name="primary") as cb:
        cb("y", "<state>y")
        cb("y", "<func>f(0, 2*i*<func>f(0, y if i > 2 else 2*y))",
                loops=(("i", 0, 5),))
        cb("<state>y", "y")

    code = create_DAGCode_with_steady_phase(cb.statements)

    rhs_function = "<func>f"

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, "ytype",
                            identifier=rhs_function,
                            input_names=("y",))
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    codegen = f.CodeGenerator(
            "selfdep",
            function_registry=freg,
            user_type_map={"ytype": f.ArrayType((100,), f.BuiltinType("real*8"))},
            timing_function="second")

    code_str = codegen(code)
    run_fortran([
        ("selfdep.f90", code_str),
        ("test_selfdep.f90", read_file("test_selfdep.f90")),
        ],
        fortran_libraries=["lapack", "blas"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
