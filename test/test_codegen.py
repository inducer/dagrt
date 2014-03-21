__copyright__ = "Copyright (C) 2014 Matt Wala"

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

import pytest
import sys

from leap.vm.language import ReturnState, CodeBuilder, TimeIntegratorCode
from leap.vm.codegen import CodeGenerator, PythonEmitter

def test_basic_codegen():
    cbuild = CodeBuilder()
    cbuild.add_and_get_ids(
        ReturnState(id='return', expression='0', component_id='<state>',
        depends_on=[]))
    cbuild.commit()
    code = TimeIntegratorCode(initialization_dep_on=[],
        instructions=cbuild.instructions, step_dep_on=['return'],
        step_before_fail=False)
    codegen = CodeGenerator()
    emitter = PythonEmitter(output='str')
    output = codegen(code, emitter)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])    
