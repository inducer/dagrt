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

from dagrt.language import YieldState
from dagrt.language import DAGCode, CodeBuilder
import dagrt.codegen.fortran as f

from utils import RawCodeBuilder

from dagrt.utils import run_fortran

#skip = pytest.mark.skipif(True, reason="not fully implemented")


def read_file(rel_path):
    from os.path import join, abspath, dirname
    path = join(abspath(dirname(__file__)), rel_path)
    with open(path, "r") as inf:
        return inf.read()


def test_basic_codegen():
    """Test whether the code generator returns a working method. The
    generated method always returns 0."""
    cbuild = RawCodeBuilder()
    cbuild.add_and_get_ids(
        YieldState(id='return', time=0, time_id='final',
                    expression=0, component_id='state',
        depends_on=[]))
    cbuild.commit()
    code = DAGCode.create_with_init_and_step(
            initialization_dep_on=[],
            instructions=cbuild.instructions, step_dep_on=['return'])
    codegen = f.CodeGenerator("simple",
            user_type_map={
                "state": f.ArrayType(
                    (200,),
                    f.BuiltinType('real (kind=8)'),)
                })
    print(codegen(code))


class MatrixInversionFailure(object):
    pass


def test_arrays_and_linalg():
    from dagrt.function_registry import base_function_registry as freg

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
        cb((), "`<builtin>print`(myzero)")
        with cb.if_("`<builtin>norm_2`(myzero) > 10**(-8)"):
            cb.raise_(MatrixInversionFailure)

    code = DAGCode.create_with_steady_state(
        cb.state_dependencies, cb.instructions)

    codegen = f.CodeGenerator(
            'arrays',
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
        fortran_options=["-llapack", "-lblas"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
