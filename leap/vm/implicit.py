"""Implicit solver utilities"""

__copyright__ = """
Copyright (C) 2014 Matt Wala
"""

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

from leap.vm.expression import EvaluationMapper


class NumpySolver(object):

    def solve(self, expression, solve_component, context, functions, guess):
        raise NotImplementedError()


class GenericNumpySolver(NumpySolver):

    def run_solver(self, func, guess):
        raise NotImplementedError()

    def solve(self, expression, solve_component, context, functions, guess):

        class FunctionWithContext(object):

            def __init__(self, expression, arg_name, context, functions):
                self.eval_mapper = EvaluationMapper(self, functions)
                self.expression = expression
                self.arg_name = arg_name
                self.context = context

            def __call__(self, arg):
                self.value = arg
                return self.eval_mapper(self.expression)

            def __getitem__(self, name):
                if name == self.arg_name:
                    return self.value
                else:
                    return self.context[name]

        func = FunctionWithContext(expression, solve_component.name,
                                   context, functions)
        return self.run_solver(func, guess)
