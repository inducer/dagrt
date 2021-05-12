"""Various usefulness"""

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

import numpy as np  # noqa
import dagrt.language as lang


# {{{ things to pass for python_method_impl

def python_method_impl_interpreter(code, **kwargs):
    from dagrt.exec_numpy import NumpyInterpreter
    return NumpyInterpreter(code, **kwargs)


def python_method_impl_codegen(code, **kwargs):
    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")
    #with open("outf.py", "w") as outf:
    #    outf.write(codegen(code))
    return codegen.get_class(code)(**kwargs)

# }}}


def execute_and_return_single_result(python_method_impl, code, initial_context=None,
                                     max_steps=1):
    if initial_context is None:
        initial_context = {}
    interpreter = python_method_impl(code, function_map={})
    interpreter.set_up(t_start=0, dt_start=0, context=initial_context)
    has_state_component = False
    for event in interpreter.run(max_steps=max_steps):
        if isinstance(event, interpreter.StateComputed):
            has_state_component = True
            state_component = event.state_component
    assert has_state_component
    return state_component


# {{{ create a DAGCode with a single, repeatedly executed phase

def create_DAGCode_with_steady_phase(statements):
    name = "main"
    phases = [
            lang.ExecutionPhase(
                name=name,
                next_phase=name,
                statements=statements)
    ]
    return lang.DAGCode.from_phases_list(phases, name)

# }}}


# {{{ create a DAGCode with an "init" phase and a "main" phase

def create_DAGCode_with_init_and_main_phases(init_statements, main_statements):
    phases = [
            lang.ExecutionPhase(
                name="init",
                next_phase="main",
                statements=init_statements),
            lang.ExecutionPhase(
                name="main",
                next_phase="main",
                statements=main_statements)
    ]
    return lang.DAGCode.from_phases_list(phases, "init")

# }}}


# {{{ low-level code building utility

class RawCodeBuilder:

    def __init__(self):
        self.id_set = set()
        self.generated_id_set = set()

        self._statements = []
        self.build_group = []

    def fresh_stmt_id(self, prefix):
        """Return an statement name that is guaranteed not to be in use and
        not to be generated in the future."""
        from pytools import generate_unique_names
        for possible_id in generate_unique_names(prefix):
            if possible_id not in self.id_set and possible_id not in \
                    self.generated_id_set:
                self.generated_id_set.add(possible_id)
                return possible_id

    def add_and_get_ids(self, *stmts):
        new_ids = []
        for stmt in stmts:
            set_attrs = {}
            if not hasattr(stmt, "id") or stmt.id is None:
                set_attrs["id"] = self.fresh_stmt_id("stmt")
            else:
                if stmt.id in self.id_set:
                    raise ValueError("duplicate ID")

            if not hasattr(stmt, "depends_on"):
                set_attrs["depends_on"] = frozenset()

            if set_attrs:
                stmt = stmt.copy(**set_attrs)

            self.build_group.append(stmt)
            new_ids.append(stmt.id)

        # For exception safety, only make state change at end.
        self.id_set.update(new_ids)
        return new_ids

    def commit(self):
        for stmt in self.build_group:
            for dep in stmt.depends_on:
                if dep not in self.id_set:
                    raise ValueError("unknown dependency id: %s" % dep)

        self._statements.extend(self.build_group)
        del self.build_group[:]

    @property
    def statements(self):
        if self.build_group:
            raise ValueError("attempted to get statements while "
                    "build group is uncommitted")

        return self._statements

# }}}

# vim: foldmethod=marker
