from __future__ import division, with_statement

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
from leap.vm.exec_numpy import NumpyInterpreter
from leap.vm.codegen import PythonCodeGenerator


@pytest.fixture(params=["interpreter", "codegen"])
def python_method_impl(request):
    kind = request.param

    def make_method_from_code(code, **kwargs):

        if kind == "interpreter":
            return NumpyInterpreter(code, **kwargs)
        elif kind == "codegen":
            codegen = PythonCodeGenerator(class_name='Method')
            return codegen.get_class(code)(**kwargs)

    return make_method_from_code


@pytest.fixture()
def execute_and_return_single_result(python_method_impl):

    def run(code):
        interpreter = python_method_impl(code, function_map={})
        interpreter.set_up(t_start=0, dt_start=0, context={})
        interpreter.initialize()
        events = [event for event in interpreter.run(t_end=0)]
        assert len(events) == 2
        assert isinstance(events[0], interpreter.StateComputed)
        assert isinstance(events[1], interpreter.StepCompleted)
        return events[0].state_component

    return run
