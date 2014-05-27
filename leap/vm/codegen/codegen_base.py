"""Provide a base class for code generators"""

from __future__ import print_function

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

from .analysis import InstructionDAGVerifier
from .dag2ir import InstructionDAGExtractor, ControlFlowGraphAssembler
from .optimization import Optimizer


class CodeGenerator(object):
    """Base class for code generation."""

    class CodeGenerationError(Exception):

        def __init__(self, errors):
            self.errors = errors

        def __str__(self):
            return 'Errors encountered in input to code generator.\n' + \
                '\n'.join(self.errors)

    def __init__(self, emitter, optimize=True, suppress_warnings=False):
        self.emitter = emitter
        self.suppress_warnings = suppress_warnings
        self.optimize = optimize

    def __call__(self, code):
        dag = code.instructions
        self.verify_dag(dag)
        extractor = InstructionDAGExtractor()
        assembler = ControlFlowGraphAssembler()
        optimizer = Optimizer()

        # Generate initialization code.
        initialization_deps = code.initialization_dep_on
        initialization = extractor(dag, initialization_deps)
        initialization_cfg = assembler(initialization, initialization_deps)
        if self.optimize:
            initialization_cfg = optimizer(initialization_cfg)
        self.emitter.emit_initialization(initialization_cfg)

        # Generate timestepper code.
        stepper_deps = code.step_dep_on
        stepper = extractor(dag, code.step_dep_on)
        stepper_cfg = assembler(stepper, stepper_deps)
        if self.optimize:
            stepper_cfg = optimizer(stepper_cfg)
        self.emitter.emit_stepper(stepper_cfg)

        return self.emitter.get_code()

    def verify_dag(self, dag):
        """Verifies that the DAG is well-formed."""
        verifier = InstructionDAGVerifier()
        errors, warnings = verifier(dag)
        if warnings and not self.suppress_warnings:
            from sys import stderr
            for warning in warnings:
                print('Warning: ' + warning, file=stderr)
        if errors:
            raise CodeGenerator.CodeGenerationError(errors)
